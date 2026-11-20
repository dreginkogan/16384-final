#!/usr/bin/env python3
"""Find the two light sticks in a ZED pair and return their end points in 3D.

    python -m libcapstone.light_stick                    # live window, 2K
    python -m libcapstone.light_stick --replay FRAMES    # re-run on saved frames
    python -m libcapstone.light_stick --no-window -v     # headless, verbose

In the window: `q` quits, `space` pauses, `s` saves the current frame, `[` and
`]` step the exposure.

There are two sticks, one red and one green, and each is found on its own. What
comes back for each is a segment: two end points in metres in the rectified
left camera frame, with an error bar on each, or a refusal saying why there is
no honest answer for this frame.

The sticks are printed covers over flashlights, so the focus decides where the
light goes. One end can clip while the middle barely registers, and which end
that is changes as the focus does. Nothing here reads brightness to decide
anything: not the threshold, and not which end is which.

Why the stick is treated as a silhouette
----------------------------------------

The stick is the worst possible target for dense stereo: emissive, saturated
and textureless, so block matching finds nothing on it and matches the wall
behind it beautifully. It is, though, trivially separable by colour. Everything
else bright in the lab -- the wall, the arm, paper -- is neutral or nearly so,
so the mask comes from how far a pixel is past neutral towards the stick's own
colour (`color_score`), with one fixed low threshold, and any clipped hot spot
is then added back where it sits against that colour.

Given the mask, the obvious next step is to match the two eyes row by row and
fit a line to the disparities. That is excellent while the stick stands up
across the image and useless once it lies along the epipolar lines, because
then a row no longer crosses the stick -- it runs along it, and the centroid of
that run is free to slide.

Here the stick is treated as what it looks like: a long silhouette. Each eye
gets an axis and two ends of its own, and disparity is then fitted along the
stick from two kinds of evidence at once --

  * the per-row centroids, as before, but weighted by how much a row's
    centroid is free to slide, which is the row's run length. A row crossing
    a near-vertical stick is worth about a pixel; the same row on a stick
    lying flat is worth fifty, and the fit discounts it accordingly instead
    of being pivoted by it.

  * the two ends, matched to each other. An end is a real feature, visible to
    both eyes whatever the angle, and it is the only thing left to go on once
    the stick lies flat.

For a line in space, disparity is exactly affine in position along the stick's
projection, so two independent observations are enough and everything above is
redundancy. Both ends in hand, nothing else is needed; this is why the flat
case works at all.

The width model is a straight line along the stick rather than a single number:
the lit silhouette narrows where the stick is dim and widens towards the
camera, and one plateau width would put a narrow end centimetres short.

Crossed sticks
--------------

Whichever stick is in front hides a piece of the other, and at a shallow angle
it cuts that one into two long wedges. The pieces are joined when the other
stick's own pixels account for the gap between them (`collinear_with`). If the
stick in front hides an *end*, what is left is a shorter stick that looks fine
in every other way, so an answer far from the stick's known length is refused.

Which end is the tip
--------------------

This is a best guess, and it is worth knowing how good a guess. The tube is the
same width all along and the hot spot goes wherever the focus sends it, so
neither says anything. Two weak cues are left: the cut end looks flatter than
the dome, and the handle and hand sit past the hilt (`hilt_evidence`). Both
depend on how the stick is held and what is behind it. A single frame gets it
wrong now and then, so `LightsaberDetector` holds each end's identity from
frame to frame and lets the evidence accumulate. `detect` alone has no history.

An end that runs off the edge of either picture is refused rather than
guessed at. The two eyes crop the scene differently -- the right one sees
less of the left of it, so the hilt leaves that eye first -- and an end at
the edge of a frame is that edge, not the end of the stick. The detection
comes back with `ok` false, `clipped` saying which end it was, and the 2D
silhouettes still attached so the overlay can show what was seen.
"""

import argparse
import contextlib
import dataclasses
import functools
import json
import logging
import os
import pathlib
import sys
import threading
import time

import cv2
import numpy as np

LOG = logging.getLogger("light_stick")


class LightStickError(RuntimeError):
    """Something this module was asked for that it cannot do.

    A plain exception rather than `SystemExit`, because most of what is here
    runs inside somebody else's program now -- a detector on a visualizer
    thread has no business ending the process. `main` turns it back into an
    exit for the command line.
    """

# Channel letters, in the order OpenCV keeps them.
CHANNELS = "bgr"


def load_stick_colors(path=None):
    """Everything that differs between the sticks, from `data/stick_colors.json`.

    One entry per stick, and the rest of this module is the same for all of
    them. `signal` and `against` define `color_score`: the weakest `signal`
    channel over the strongest `against` channel. `clip_channel` is the one that
    clips first in a hot spot, `threshold` the score a stick pixel reaches, and
    the two colours (RGB hex) are what the stick is drawn in.

    The threshold is absolute, and low, on purpose: the flashlight's focus moves
    the light along the stick, so one end can clip while the middle scores 30,
    and any fraction of the peak then cuts the stick in half.
    """
    if path is None:
        path = pathlib.Path(__file__).resolve().parent / "data" / "stick_colors.json"
    try:
        settings = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError) as error:
        raise LightStickError(f"could not read the stick colours {path}: {error}")

    def rgb(text):
        value = int(text.lstrip("#"), 16)
        return value, (value & 0xFF, (value >> 8) & 0xFF, value >> 16)

    sticks = {}
    for color, entry in settings.items():
        try:
            sticks[color] = {
                "signal": [CHANNELS.index(c) for c in entry["signal"]],
                "against": [CHANNELS.index(c) for c in entry["against"]],
                "clip_channel": CHANNELS.index(entry["clip_channel"]),
                "threshold": int(entry["threshold"]),
                "draw_color": rgb(entry["draw_color"])[0],
                "overlay_edge": rgb(entry["overlay_color"])[1],  # BGR, for OpenCV
            }
        except (KeyError, TypeError, ValueError) as error:
            raise LightStickError(f"{path}: bad settings for {color!r}: {error!r}")
        if not sticks[color]["signal"] or not sticks[color]["against"]:
            raise LightStickError(f"{path}: {color!r} needs both signal and against channels")
    return sticks


STICKS = load_stick_colors()
COLORS = tuple(STICKS)

# The hot spot can clip towards white, so it is found by brightness -- but only
# where it sits against the stick's colour, which keeps the ceiling lights out.
BRIGHT_THRESHOLD = 250
CORE_DILATE = 9

MIN_PIXELS = 400  # a component smaller than this is noise or a reflection
MIN_SPAN = 60  # pixels of blade, below which there is nothing to fit
MIN_ASPECT = 4.0  # length over width; the stick is nearer 9
# A piece of mask an occluder split off is at least this much of the piece
# that was kept. Anything smaller is a stray and is left alone -- unless the
# other stick is what lies between them, which is its own evidence.
COLLINEAR_SHARE = 0.08
# When the other stick is what cut this one, the two pieces are wedges and each
# one's own axis leans, so they are held to this many half-widths off each
# other's line rather than the usual 1.2.
CROSSED_REACH = 3.0
# Of the way between two pieces, the share the other stick has to account for.
CROSSED_SHARE = 0.5

# What the stick measures end to end, off the saved bench frames, and how far
# from that an answer is still believed. An end hidden behind the other stick
# gives a short stick that otherwise looks perfectly healthy.
STICK_LENGTH = 0.255
MAX_LENGTH_ERROR = 0.04

# Telling the hilt from the tip. Neither brightness nor width can: the focus
# moves the hot spot to either end, and the tube is 30 mm across all the way
# along. Two weak cues are left, each put in units of its own typical size:
#   * the cut end is flatter than the dome, read a quarter radius in;
#   * the handle and the hand sit just past the hilt, and past the tip is
#     whatever is behind the stick. Grey levels darker than beside the end.
FLATNESS_SCALE = 0.15
SHADOW_SCALE = 3.0
# Each frame's vote counts this much of the running one, on the detector.
VOTE_DECAY = 0.95

# Rows whose two eyes disagree this much about how many blade pixels they hold
# are not looking at the same piece of blade, so their centroids are not a
# disparity. Likewise rows holding far less than a typical row, which are the
# ones clipping an end.
MAX_COUNT_RATIO = 1.6
MIN_COUNT_SHARE = 0.4

# How far a row centroid can slide, per pixel of the row's run across the
# blade. A row crossing a steep blade pins the centroid; as the blade flattens
# the run lengthens and the same centroid is free to move along it, so this
# turns the blade's angle into an honest error bar instead of a hard cutoff.
ROW_SLIDE = 0.02
ROW_FLOOR = 0.5  # pixels, the best a row centroid is ever worth
# Rows are correlated -- a mask biased at one row is biased at the next -- so
# averaging them does not beat their common error down. They share the weight
# of this many independent observations between them however many there are.
ROW_BUDGET = 8.0

# How far apart the two eyes can put the same end of the blade, along the
# blade. Only the part of that along the x axis is a disparity error, which is
# why a steep blade is barely troubled by it and a flat one is troubled fully.
END_SLIDE = 3.0
END_FLOOR = 1.0

# An end this close to the edge of its image is the edge of the picture rather
# than the end of the stick.
BORDER_MARGIN = 25

# Depth error at an end, from the disparity fit's own covariance, past which
# the answer is not worth handing to the arm.
MAX_DEPTH_ERROR = 0.15

# On rectified images the two eyes should differ only horizontally.
MAX_EYE_DY = 3.0
EYE_CHECK_SCALE = 0.125


@dataclasses.dataclass
class Blade:
    """One eye's view of a stick: an axis, two ends and a width along it."""

    mask: np.ndarray
    centre: np.ndarray  # of the silhouette, in pixels
    along: np.ndarray  # unit vector down the blade
    across: np.ndarray  # unit vector across it
    t0: float  # one end, as a distance along `along` from `centre`
    t1: float  # the other; `align` makes the two eyes agree on which is which
    p0: np.ndarray  # those two as pixel coordinates
    p1: np.ndarray
    w0: float  # modelled silhouette width at each end, in pixels
    w1: float
    profile: tuple  # (positions, measured widths) for the overlay
    flat0: float = 0.0  # silhouette width just inside each end, over the tube's
    flat1: float = 0.0

    @property
    def span(self):
        return self.t1 - self.t0

    @property
    def run(self):
        """How far a single image row runs along the blade, in pixels.

        One pixel of blade width if the blade stands straight up, the whole
        blade if it lies flat. This is the number that says whether row
        matching means anything in this frame.
        """
        return 0.5 * (self.w0 + self.w1) / max(abs(self.along[1]), 1e-6)


@dataclasses.dataclass
class Detection:
    """What one frame yielded for one stick, whether or not it found it."""

    ok: bool
    reason: str = ""
    color: str = ""  # which stick this is about
    ends: np.ndarray = None  # (2, 3) metres, rectified left camera frame: tip, then hilt
    ends_px: np.ndarray = None  # (2, 2) the same two points in the left image
    clipped: tuple = (False, False)  # that end is the frame's edge, not the stick's
    length: float = 0.0  # metres between the two ends
    direction: np.ndarray = None  # unit vector from the tip to the hilt
    disparity: tuple = (0.0, 0.0)  # pixels, at each end
    depth_error: tuple = (0.0, 0.0)  # metres of one sigma, at each end
    diameter: tuple = (0.0, 0.0)  # metres of lit silhouette across, at each end
    hilt_evidence: float = 0.0  # how much this frame says `ends` are the right way round
    residual: float = 0.0  # pixels, of the disparity fit
    rows_used: int = 0
    row_sigma: float = 0.0  # pixels a row centroid was worth in this frame
    left: Blade = None
    right: Blade = None
    timings: dict = dataclasses.field(default_factory=dict)

    def flipped(self):
        """The same detection with its tip and hilt the other way round."""
        return dataclasses.replace(
            self,
            ends=self.ends[::-1],
            ends_px=self.ends_px[::-1],
            clipped=self.clipped[::-1],
            direction=-self.direction,
            disparity=self.disparity[::-1],
            depth_error=self.depth_error[::-1],
            diameter=self.diameter[::-1],
            hilt_evidence=-self.hilt_evidence,
        )

    @property
    def midpoint(self):
        return None if self.ends is None else self.ends.mean(axis=0)

    @property
    def distance(self):
        return 0.0 if self.ends is None else float(np.linalg.norm(self.midpoint))


@dataclasses.dataclass
class Options:
    """Everything `detect` reads, with the tuned defaults above.

    `detect` used to be handed the command line\'s own `argparse.Namespace`,
    which worked only as long as the only caller was the command line. These
    are the values it actually looks at; the CLI builds one of these from
    its arguments, and nothing else has to know the flags exist.
    """

    dilate: int = CORE_DILATE
    bright_threshold: int = BRIGHT_THRESHOLD
    min_pixels: int = MIN_PIXELS
    color_threshold: int = None  # overrides every stick's own, if set
    max_depth_error: float = MAX_DEPTH_ERROR

    @classmethod
    def from_namespace(cls, args):
        """The options out of a parsed command line."""
        return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})


def color_score(image, color):
    """How much of this stick's colour a pixel has beyond neutral.

    The weakest of the stick's `signal` channels over the strongest of its
    `against` channels. Under the bench's white balance the green stick comes
    out with G and B about equal, which is why it asks for both, and the wall's
    purple has plenty of B but little G.

    Blurred, because this is a difference of channels and so carries the noise
    of all three, and on a dark frame at gain that speckles the mask; a few
    pixels of blur costs nothing on an object this size.
    """
    check_color(color)
    channels = cv2.split(image)
    # Saturating uint8 arithmetic throughout: a pixel with less of the colour
    # than neutral is of no interest, so clamping it at zero costs nothing and
    # keeps this off the int16 path, which at 2208x1242 is worth having.
    signal = functools.reduce(cv2.min, [channels[i] for i in STICKS[color]["signal"]])
    against = functools.reduce(cv2.max, [channels[i] for i in STICKS[color]["against"]])
    return cv2.GaussianBlur(cv2.subtract(signal, against), (5, 5), 0)


def color_threshold(color, options):
    """The score a pixel of this stick reaches: its own, unless overridden."""
    if options.color_threshold is not None:
        return options.color_threshold
    return STICKS[color]["threshold"]


def fill_holes(mask):
    """Close off anything enclosed by the silhouette.

    A hot spot clipped white scores nothing on colour, and one that the
    brightness test misses leaves a hole that must not read as an end.
    """
    # Flooded from a one-pixel border rather than from a corner, because the
    # blade can touch the corner of the box it is cropped to and would then
    # be flooded as background.
    height, width = mask.shape
    flooded = np.zeros((height + 2, width + 2), np.uint8)
    flooded[1:-1, 1:-1] = mask
    cv2.floodFill(flooded, np.zeros((height + 4, width + 4), np.uint8), (0, 0), 1)
    return (mask | (1 - flooded[1:-1, 1:-1])).astype(np.uint8)


def threshold_blade(score, image, color, options):
    """The stick's colour, plus any clipped hot spot sitting against it.

    The hot spot is read off this stick's own channel, so a clipped green stick
    crossing a red one is not taken in as part of the red one.
    """
    mask = (score > color_threshold(color, options)).astype(np.uint8)
    near = cv2.dilate(mask, np.ones((options.dilate, options.dilate), np.uint8))
    clip = STICKS[color]["clip_channel"]
    mask |= (image[:, :, clip] >= options.bright_threshold).astype(np.uint8) & near
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def pick_blade(mask, options, occluder=None):
    """The one component that is the stick, plus any piece an occluder split off.

    `occluder` is a function giving the other stick's pixels, called only if
    there is a second piece to explain, which most frames there is not.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    # Big *and* long: the amber bench light in these frames clears the colour
    # test and is the second largest thing in the mask most nights, but it is
    # a blob and the blade is a streak.
    best, score = None, 0.0
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if area < options.min_pixels:
            continue
        box = (stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT])
        shape = max(box) / max(min(box), 1)
        if area * shape > score:
            best, score = index, area * shape
    if best is None:
        return None

    chosen = labels == best
    # Nothing else is big enough to be a piece of it, which is the usual case
    # and worth checking before fitting an axis to look for one.
    if not any(stats[i, cv2.CC_STAT_AREA] >= options.min_pixels for i in range(1, count) if i != best):
        return chosen

    # Twice, because an occluder can leave three pieces and the middle one
    # has to be taken in before the far one is within reach.
    hidden = occluder() if occluder is not None else None
    taken = {best}
    for _ in range(2):
        found = collinear_with(chosen, labels, stats, centroids, taken, options, hidden)
        if not found:
            break
        for index in found:
            chosen |= labels == index
            taken.add(index)
    return chosen


def blade_mask(image, color, options):
    """Threshold out one stick and return its silhouette, holes filled, and the box it is in."""
    score = color_score(image, color)

    # Everything after the first threshold happens inside the box the coloured
    # pixels fall in, which is usually a small part of the frame. Dilating
    # and labelling whole 2208x1242 frames, two eyes and two colours of them,
    # is otherwise most of the time this detector spends, and more than the
    # camera gives it between frames.
    hot = score > color_threshold(color, options)
    rows, columns = np.flatnonzero(hot.any(1)), np.flatnonzero(hot.any(0))
    if rows.size == 0:
        return None, None
    pad = options.dilate + 8
    height, width = score.shape
    top, bottom = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, height)
    left, right = max(columns[0] - pad, 0), min(columns[-1] + pad + 1, width)
    box = (slice(top, bottom), slice(left, right))

    def other_stick():
        hidden = np.zeros(score[box].shape, bool)
        for other in COLORS:
            if other != color:
                hidden |= color_score(image[box], other) > color_threshold(other, options)
        return hidden

    mask = threshold_blade(score[box], image[box], color, options)
    chosen = pick_blade(mask, options, other_stick)
    if chosen is None:
        return None, None

    mask = np.zeros(score.shape, np.uint8)
    mask[box] = fill_holes(chosen.astype(np.uint8))
    return mask, box


def crossed_between(hidden, labels, start, end):
    """Is what lies between these two points mostly the other stick?

    Only the points that belong to no piece of this stick count: what is being
    asked is what the gap is made of.
    """
    samples = int(np.hypot(*(end - start))) + 2
    xs = np.linspace(start[0], end[0], samples).round().astype(int)
    ys = np.linspace(start[1], end[1], samples).round().astype(int)
    gap = labels[ys, xs] == 0
    return bool(gap.any() and hidden[ys[gap], xs[gap]].mean() >= CROSSED_SHARE)


def collinear_with(chosen, labels, stats, centroids, taken, options, hidden=None):
    """Other pieces of mask that lie on the same stick as the one picked.

    Anything crossing in front of the stick -- the other stick, a cable, the
    other hand, the arm itself -- cuts the silhouette in two, and taking the
    larger half alone reports a stick half the length it is.

    The other stick is the one occluder this can recognise, and a crossing at a
    shallow angle cuts this one into two long wedges whose axes lean away from
    each other. So where `hidden`, the other stick's pixels, fills the gap
    between two pieces, a small piece counts and the line is drawn more loosely.


    The bar is deliberately high: a piece has to be a sizeable fraction of
    what was kept, sit on its axis, and lie within reach of its ends. Small
    stray blobs near the blade's line are what this used to let in, and they
    are the worst thing it could let in, because the two eyes do not see the
    same ones -- a speck absorbed on the left and not on the right moves that
    eye's end and nothing else, which is a disparity error dressed up as a
    longer stick.
    """
    ys, xs = (axis.astype(float) for axis in np.nonzero(chosen))
    centre, along, across = blade_axis(xs, ys)
    t = (xs - centre[0]) * along[0] + (ys - centre[1]) * along[1]
    offset = (xs - centre[0]) * across[0] + (ys - centre[1]) * across[1]
    halfwidth = max(np.percentile(np.abs(offset), 95), 2.0)
    low, high = t.min(), t.max()
    # A gap as long as the piece we kept: half a blade is exactly what an
    # occluder across the middle leaves behind.
    reach = (high - low) + 100.0

    smallest = max(options.min_pixels, COLLINEAR_SHARE * int(chosen.sum()))
    found = []
    for index in range(1, len(stats)):
        if index in taken or stats[index, cv2.CC_STAT_AREA] < options.min_pixels:
            continue
        dx, dy = centroids[index] - centre
        if not low - reach < dx * along[0] + dy * along[1] < high + reach:
            continue
        off_axis = abs(dx * across[0] + dy * across[1])
        if stats[index, cv2.CC_STAT_AREA] >= smallest and off_axis <= 1.2 * halfwidth:
            found.append(index)
        elif (
            hidden is not None
            and off_axis <= CROSSED_REACH * halfwidth
            and crossed_between(hidden, labels, centre, centroids[index])
        ):
            found.append(index)
    return found


def blade_axis(xs, ys, iterations=4):
    """Total-least-squares axis, re-fitted after trimming what sits off to the side."""
    keep = np.ones(len(xs), bool)
    centre = along = across = None
    for _ in range(iterations):
        covariance = np.cov(np.vstack([xs[keep], ys[keep]]))
        centre = np.array([xs[keep].mean(), ys[keep].mean()])
        along = np.linalg.eigh(covariance)[1][:, -1]
        across = np.array([-along[1], along[0]])
        offset = (xs - centre[0]) * across[0] + (ys - centre[1]) * across[1]
        keep = np.abs(offset) < 3 * np.percentile(np.abs(offset[keep]), 95)
    return centre, along, across


def width_profile(t, offset, bin_px=4.0):
    """Silhouette width against position along the blade.

    The extent across rather than the pixel count, so that a hole left over in
    the mask costs nothing and a frayed edge costs little.
    """
    low, high = t.min(), t.max()
    bins = max(8, int(round((high - low) / bin_px)))
    index = np.clip(((t - low) / (high - low) * bins).astype(int), 0, bins - 1)
    upper = np.full(bins, -np.inf)
    lower = np.full(bins, np.inf)
    np.maximum.at(upper, index, offset)
    np.minimum.at(lower, index, offset)
    filled = np.isfinite(upper)
    positions = low + (np.arange(bins) + 0.5) * (high - low) / bins
    return positions[filled], (upper - lower)[filled]


def taper_model(positions, widths):
    """A straight line through the width profile, which is not constant.

    The tube itself is, at about 30 mm, but the lit silhouette is narrower
    where the focus leaves it dim and wider towards the camera. Calling that
    one number puts the narrow end several centimetres out.
    """
    low, high = positions[0], positions[-1]
    inner = (positions > low + 0.15 * (high - low)) & (positions < high - 0.15 * (high - low))
    if inner.sum() < 4:
        inner = np.ones(len(positions), bool)
    x, y = positions[inner], widths[inner]
    slope, intercept = np.polyfit(x, y, 1)
    for _ in range(3):
        residual = y - (slope * x + intercept)
        spread = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        keep = np.abs(residual) < 2.5 * spread + 1e-6
        if keep.sum() < 4:
            break
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
    return slope, intercept


def find_ends(positions, widths, slope, intercept):
    """Outermost place the silhouette falls to half the width it should have there.

    Half of the *modelled* width, so that the tapered end is not cut short and
    a hole in the middle is not mistaken for the end.
    """
    model = np.maximum(slope * positions + intercept, 1.0)
    over = widths >= 0.5 * model

    def crossing(order):
        previous = None
        for i in order:
            if over[i]:
                if previous is None:
                    return positions[i]
                before = widths[previous] - 0.5 * model[previous]
                after = widths[i] - 0.5 * model[i]
                fraction = -before / (after - before)
                return positions[previous] + fraction * (positions[i] - positions[previous])
            previous = i
        return None

    count = len(positions)
    return crossing(range(count)), crossing(range(count - 1, -1, -1))


def see_blade(image, color, options):
    """One eye's view of one stick: mask, axis, ends, width. None if it is not there."""
    mask, box = blade_mask(image, color, options)
    if mask is None:
        return None

    # Within the box the mask was built in, so that this reads a corner of
    # the frame rather than all of it, then back into image coordinates.
    ys, xs = (axis.astype(float) for axis in np.nonzero(mask[box]))
    ys += box[0].start
    xs += box[1].start
    centre, along, across = blade_axis(xs, ys)
    t = (xs - centre[0]) * along[0] + (ys - centre[1]) * along[1]
    offset = (xs - centre[0]) * across[0] + (ys - centre[1]) * across[1]

    positions, widths = width_profile(t, offset)
    slope, intercept = taper_model(positions, widths)
    t0, t1 = find_ends(positions, widths, slope, intercept)
    if t0 is None or t1 is None or t1 - t0 < MIN_SPAN:
        return None

    w0 = max(slope * t0 + intercept, 1.0)
    w1 = max(slope * t1 + intercept, 1.0)
    if (t1 - t0) / max(w0, w1) < MIN_ASPECT:
        return None

    # t0 is the narrow end by convention, so that the two eyes agree on which
    # end is which without having to compare them.
    if w0 > w1:
        t0, t1, w0, w1 = t1, t0, w1, w0
        along, across = -along, -across
        t, offset, positions = -t, -offset, -positions
        t0, t1 = -t0, -t1
        slope = -slope
        order = np.argsort(positions)
        positions, widths = positions[order], widths[order]

    middle = (positions > t0 + 0.25 * (t1 - t0)) & (positions < t1 - 0.25 * (t1 - t0))
    tube = max(float(np.median(widths[middle])) if middle.any() else max(w0, w1), 1.0)

    return Blade(
        mask=mask,
        centre=centre,
        along=along,
        across=across,
        t0=t0,
        t1=t1,
        p0=centre + t0 * along,
        p1=centre + t1 * along,
        w0=w0,
        w1=w1,
        profile=(positions, widths),
        flat0=end_flatness(-t, offset, tube),
        flat1=end_flatness(t, offset, tube),
    )


def end_flatness(t, offset, tube):
    """Silhouette width a quarter radius in from the far end of `t`, over the tube's.

    A dome has narrowed to two thirds of the tube there. A cut end seen side on
    has not narrowed at all.
    """
    end = np.percentile(t, 99.95)
    inside = 0.25 * 0.5 * tube
    band = (t <= end - inside) & (t > end - inside - 3.0)
    return float(np.ptp(offset[band]) / tube) if band.any() else 0.0


def end_shadow(image, blade, point, outward):
    """How much darker it is just past this end than beside it, in grey levels.

    Medians, so the other stick wandering through either patch costs little.
    """
    radius = 0.25 * (blade.w0 + blade.w1)
    height, width = image.shape[:2]

    def grey(alongs, acrosses):
        t, o = np.meshgrid(alongs, acrosses)
        xs = np.round(point[0] + outward * (t * blade.along[0]) + o * blade.across[0]).astype(int)
        ys = np.round(point[1] + outward * (t * blade.along[1]) + o * blade.across[1]).astype(int)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not inside.any():
            return np.nan
        return float(np.median(image[ys[inside], xs[inside]].mean(axis=1)))

    past = grey(np.arange(0.3 * radius, 3.0 * radius, 3.0), np.arange(-radius, radius, 3.0))
    beside = grey(
        np.arange(-3.0 * radius, 3.0 * radius, 3.0),
        np.concatenate([np.arange(1.6 * radius, 3.5 * radius, 3.0),
                        -np.arange(1.6 * radius, 3.5 * radius, 3.0)]),
    )
    shadow = beside - past
    return 0.0 if np.isnan(shadow) else shadow


def hilt_evidence(image, blade):
    """Positive if this eye's p1 looks like the hilt, negative if its p0 does."""
    flatness = blade.flat1 - blade.flat0
    shadow = end_shadow(image, blade, blade.p1, 1.0) - end_shadow(image, blade, blade.p0, -1.0)
    return flatness / FLATNESS_SCALE + shadow / SHADOW_SCALE


def align(left, right):
    """Point the right eye's blade the same way as the left eye's.

    Each eye picks its own direction from which of its ends looks narrower,
    which is right nearly always and catastrophic the once it is not -- the
    two eyes then pair each end with the other one and the disparities come
    back hundreds of pixels apart. The two views of the blade are within a
    degree or two of each other, so their dot product settles it outright.
    """
    if np.dot(left.along, right.along) >= 0:
        return right
    return dataclasses.replace(
        right,
        along=-right.along,
        across=-right.across,
        t0=-right.t1,
        t1=-right.t0,
        p0=right.p1,
        p1=right.p0,
        w0=right.w1,
        w1=right.w0,
        profile=(-right.profile[0][::-1], right.profile[1][::-1]),
        flat0=right.flat1,
        flat1=right.flat0,
    )


def row_disparities(left, right):
    """Per-row centroid disparity, kept only where both eyes cross the same blade."""
    rows = np.flatnonzero(left.mask.any(1) & right.mask.any(1))
    if rows.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    columns = np.arange(left.mask.shape[1])
    count_left = left.mask[rows].sum(1)
    count_right = right.mask[rows].sum(1)
    centre_left = (left.mask[rows] * columns).sum(1) / np.maximum(count_left, 1)
    centre_right = (right.mask[rows] * columns).sum(1) / np.maximum(count_right, 1)

    typical = np.median(np.concatenate([count_left, count_right]))
    keep = (
        (count_left > 0)
        & (count_right > 0)
        & (np.maximum(count_left, count_right) <= MAX_COUNT_RATIO * np.minimum(count_left, count_right))
        & (count_left >= MIN_COUNT_SHARE * typical)
        & (count_right >= MIN_COUNT_SHARE * typical)
    )
    return (
        rows[keep].astype(float),
        centre_left[keep],
        centre_left[keep] - centre_right[keep],
    )


def fit_disparity(observations, iterations=5):
    """Weighted straight line through disparity against position along the blade.

    Returns the coefficients, the covariance of them, and the residual. For a
    straight blade this line is not an approximation: disparity really is
    affine in position along the projection of a line in space, so the two
    ends alone already fix it and the rows are redundancy.
    """
    positions = np.concatenate([o[0] for o in observations])
    disparity = np.concatenate([o[1] for o in observations])
    weight = np.concatenate([o[2] for o in observations])

    design = np.vstack([positions, np.ones_like(positions)]).T
    coefficients = np.zeros(2)
    for _ in range(iterations + 1):
        root = np.sqrt(weight)
        coefficients, *_ = np.linalg.lstsq(design * root[:, None], disparity * root, rcond=None)
        residual = disparity - design @ coefficients
        spread = 1.4826 * np.median(np.abs(residual)) + 1e-9
        trimmed = weight * (np.abs(residual) < 3 * spread)
        if trimmed.sum() <= 0:
            break
        weight = trimmed

    normal = design.T @ (design * weight[:, None])
    covariance = np.linalg.inv(normal) if np.linalg.det(normal) > 1e-12 else np.full((2, 2), np.inf)
    residual = disparity - design @ coefficients
    used = weight > 0

    # The error bars above are only as honest as the weights that went in. If
    # the observations disagree by more than those weights allow, that is the
    # fit telling us the weights were optimistic, so widen them to suit. This
    # is what makes a blade lying along the epipolar lines report itself as
    # uncertain rather than confidently wrong.
    freedom = max(float(weight[used].sum() / weight[used].max()) - 2, 1.0)
    chi2 = float((weight[used] * residual[used] ** 2).sum() / freedom)
    covariance = covariance * max(chi2, 1.0)

    scatter = float(np.median(np.abs(residual[used]))) if used.any() else float("nan")
    return coefficients, covariance, scatter, int(used.sum())


def triangulate(calibration, points_px, disparities):
    """Rectified pixels plus disparity into metres in the left camera frame."""
    _, _, P1, P2, _ = calibration.rectify()
    left = np.asarray(points_px, float).T
    right = np.vstack([left[0] - np.asarray(disparities, float), left[1]])
    homogeneous = cv2.triangulatePoints(P1, P2, left, right)
    return (homogeneous[:3] / homogeneous[3]).T


def clipped_end(blade, point):
    """Is this end the edge of the picture rather than the end of the stick?"""
    height, width = blade.mask.shape
    return bool(
        point[0] < BORDER_MARGIN
        or point[0] > width - 1 - BORDER_MARGIN
        or point[1] < BORDER_MARGIN
        or point[1] > height - 1 - BORDER_MARGIN
    )


def detect_all(left_image, right_image, calibration, options=None):
    """Both sticks in a rectified pair: `{"red": Detection, "green": Detection}`."""
    return {
        color: detect(left_image, right_image, calibration, color, options)
        for color in COLORS
    }


def detect(left_image, right_image, calibration, color, options=None):
    """Find one stick in a rectified pair and place both of its ends in 3D.

    `left_image` and `right_image` are rectified BGR frames from the two eyes,
    `calibration` the `StereoCalibration` they were rectified with, `color` is
    "red" or "green", and `options` an `Options` (the tuned defaults, if left
    out).
    """
    options = options if options is not None else Options()
    timings = {}

    started = time.perf_counter()
    left = see_blade(left_image, color, options)
    right = see_blade(right_image, color, options)
    timings["mask"] = 1000 * (time.perf_counter() - started)
    if left is None:
        return Detection(False, f"no {color} stick in the left eye", color=color, timings=timings)
    if right is None:
        return Detection(False, f"no {color} stick in the right eye", color=color, timings=timings)
    right = align(left, right)

    # An end that ran off the edge of either picture is that edge, not the
    # end of the stick, and the two eyes crop the scene differently -- the
    # right one sees less of the left of it, so the hilt leaves that eye
    # first. Say so and stop: there is no honest segment to report.
    clipped = tuple(
        clipped_end(left, point) or clipped_end(right, other)
        for point, other in ((left.p0, right.p0), (left.p1, right.p1))
    )
    if any(clipped):
        which = "both ends" if all(clipped) else "an end"
        return Detection(
            False,
            f"{which} of the {color} stick runs off the edge of a picture",
            color=color, clipped=clipped, left=left, right=right, timings=timings,
        )

    started = time.perf_counter()
    rows, centres, row_disparity = row_disparities(left, right)
    row_sigma = ROW_FLOOR + ROW_SLIDE * left.run
    observations = []
    if rows.size:
        position = (centres - left.centre[0]) * left.along[0] + (rows - left.centre[1]) * left.along[1]
        share = min(1.0, ROW_BUDGET / rows.size)
        observations.append((position, row_disparity, np.full(rows.size, share / row_sigma**2)))

    # An end mismatched between the eyes slides along the blade, so only the
    # part of that along x is a disparity error.
    end_sigma = END_FLOOR + END_SLIDE * abs(left.along[0])
    ends_px = np.vstack([left.p0, left.p1])
    for t, point_left, point_right in ((left.t0, left.p0, right.p0), (left.t1, left.p1, right.p1)):
        observations.append(
            (np.array([t]), np.array([point_left[0] - point_right[0]]), np.array([1 / end_sigma**2]))
        )

    (slope, intercept), covariance, scatter, _ = fit_disparity(observations)
    timings["match"] = 1000 * (time.perf_counter() - started)

    disparities = np.array([slope * left.t0 + intercept, slope * left.t1 + intercept])
    if np.any(disparities <= 0):
        return Detection(False, "the disparity fit came out negative", color=color, timings=timings)

    started = time.perf_counter()
    points = triangulate(calibration, ends_px, disparities)
    timings["triangulate"] = 1000 * (time.perf_counter() - started)

    # The fit's own covariance, turned into metres at each end.
    focal, baseline = calibration.focal_length, calibration.baseline
    variance = np.array(
        [[t, 1.0] @ covariance @ [t, 1.0] for t in (left.t0, left.t1)]
    )
    sigma = np.sqrt(np.maximum(variance, 0.0))
    depth_error = points[:, 2] ** 2 / (focal * baseline) * sigma

    widths = np.array([left.w0, left.w1])
    diameter = widths * points[:, 2] / focal

    # Tip first, on this frame's evidence alone. That is a best guess and no
    # more: `LightsaberDetector` holds the two ends apart over time, which is
    # what makes the label steady.
    evidence = hilt_evidence(left_image, left)
    if evidence < 0:
        order = [1, 0]
        points, ends_px = points[order], ends_px[order]
        disparities, depth_error = disparities[order], depth_error[order]
        diameter = diameter[order]
        clipped = clipped[::-1]
        evidence = -evidence

    separation = points[1] - points[0]
    length = float(np.linalg.norm(separation))
    detection = Detection(
        ok=True,
        color=color,
        ends=points,
        ends_px=ends_px,
        clipped=clipped,
        length=length,
        direction=separation / max(length, 1e-9),
        disparity=tuple(float(d) for d in disparities),
        depth_error=tuple(float(e) for e in depth_error),
        diameter=tuple(float(d) for d in diameter),
        hilt_evidence=float(evidence),
        residual=scatter,
        rows_used=int(rows.size),
        row_sigma=float(row_sigma),
        left=left,
        right=right,
        timings=timings,
    )
    if abs(length - STICK_LENGTH) > MAX_LENGTH_ERROR:
        detection.ok = False
        detection.reason = (
            f"the {color} stick comes out {length:.3f} m long, not {STICK_LENGTH:.3f} m "
            "-- something is hiding an end of it"
        )
    elif max(depth_error) > options.max_depth_error:
        detection.ok = False
        detection.reason = (
            f"the ends are only good to {max(depth_error):.3f} m -- the blade is "
            f"{np.degrees(np.arctan2(abs(left.along[1]), abs(left.along[0]))):.0f} deg "
            "off the epipolar lines and there is little to match along it"
        )
    return detection


class ReplayCamera:
    """A folder of saved frames, pretending to be a camera.

    What `--replay` reads, and what makes the live path runnable on a machine
    with no ZED at all. The saved frames are already rectified, which is why
    `read_rectified` is the method the camera protocol names: this one just
    hands the next pair back.
    """

    def __init__(self, folder, serial=None, conf=None, resolution="2K",
                 loop=True, fps=10.0):
        from xarm7_lib.calibration import StereoCalibration

        self.folder = pathlib.Path(folder)
        self._frames = sorted(self.folder.glob("*_rect_left.png"))
        if not self._frames:
            raise LightStickError(f"no *_rect_left.png under {self.folder}")

        if conf:
            self._calibration = StereoCalibration.from_file(
                conf, resolution=resolution, serial=serial
            )
        elif serial:
            self._calibration = StereoCalibration.from_serial(serial, resolution=resolution)
        else:
            raise LightStickError("a replay needs --conf or --serial to calibrate with")

        self.serial = serial or getattr(self._calibration, "serial", None)
        self.resolution = resolution
        self._loop = bool(loop)
        self._period = 1.0 / fps if fps else None
        self._index = 0
        self._next = None

    @property
    def calibration(self):
        return self._calibration

    @property
    def exhausted(self):
        return not self._loop and self._index >= len(self._frames)

    def read_rectified(self):
        """The next saved pair, paced so a replay looks like a camera."""
        if self._index >= len(self._frames):
            if not self._loop:
                raise LightStickError(f"replay of {self.folder} is finished")
            self._index = 0

        if self._period is not None:
            now = time.monotonic()
            if self._next is not None and now < self._next:
                time.sleep(self._next - now)
            self._next = max(time.monotonic(), (self._next or time.monotonic())) + self._period

        left_path = self._frames[self._index]
        self._index += 1
        right_path = left_path.with_name(left_path.name.replace("_rect_left", "_rect_right"))

        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        if left is None or right is None:
            raise LightStickError(f"could not read the pair for {left_path.name}")
        self.name = left_path.name.replace("_rect_left.png", "")
        return left, right

    def close(self):
        pass


# The table tag. The detector streams at an exposure chosen for the sticks, and
# at that exposure a paper tag on the table is usually too dark to decode, so it
# is looked for in every frame and checked whenever it turns up: lit by a phone
# flashlight when somebody suspects the camera has been knocked, or all the time
# on a bench bright enough to see it.
#
# Each frame looks in a window around where it was last seen, which costs about
# a millisecond an eye. While it is not in that window, the whole frame is
# searched once every `TAG_SEARCH_PERIOD`, so a camera knocked further than the
# window is still found -- and refused, rather than reported as not in view.
TAG_WINDOW = 250  # px past the tag's last corners, each way
TAG_SEARCH_PERIOD = 1.0  # s between whole-frame searches while it is not in the window
# A sighting after this long without one is a new look at the tag -- somebody
# has just lit it -- and its result is said out loud, whatever it is.
TAG_QUIET = 10.0  # s


def _say(text):
    print(f"[libcapstone] {text}", file=sys.stderr, flush=True)


class TagWatch:
    """Keeps a bench's hand-eye calibration honest against its table tag.

    Fed the rectified pairs the detector already grabs. Every `LOOK_FRAMES`
    sightings, the tag's median corners are fitted against the reference
    recorded at calibration time, with the library's own `fit_camera_shift` and
    thresholds, and `handeye`
    becomes the filed calibration or a corrected one. However far the camera
    has moved, a correction is used: it is a stopgap until the bench is
    recalibrated, and the warning says how far off to expect it to be.

    A new correction replaces the one in use only when it differs from it by
    more than the library's own "has it moved" thresholds, so a camera sitting
    near a threshold does not flip between two transforms from one look to the
    next.

    Nothing is saved. Every run starts from the calibration as filed, and the
    tag corrects it again if it is lit.
    """

    def __init__(self, filed, resolution):
        from xarm7_lib.tag_recovery import tag_detector

        self.filed = filed
        self.reference = filed.tag
        self.handeye = filed
        self.last_shift = None  # the latest fit, whether or not it was adopted
        self.corners = None  # (4, 2) left-eye corners at the last sighting
        self.seen_at = None  # time.monotonic() of the last sighting

        self._resolution_ok = resolution == self.reference.resolution
        self._detector = tag_detector(self.reference.dictionary)
        self._expected = (np.asarray(self.reference.left, float),
                          np.asarray(self.reference.right, float))
        self._full_at = -np.inf
        self._announce = True  # say the result of the next check
        self._batch = ([], [], 0)
        if not self._resolution_ok:
            _say(
                f"the table tag was recorded at {self.reference.resolution} and this "
                f"camera is streaming at {resolution}, so the camera is not being "
                "checked against it."
            )

    def status(self):
        """One line on the tag, for the live window."""
        tag = f"table tag {self.reference.tag_id}"
        if not self._resolution_ok:
            return f"{tag}: not checked at this resolution"
        seen = (
            "not in view" if self.seen_at is None
            else f"seen {time.monotonic() - self.seen_at:.1f} s ago"
        )
        shift = self.handeye.shift
        state = "camera as calibrated" if shift is None else f"corrected for {shift.describe()}"
        return f"{tag}: {seen}, {state}"

    def feed(self, left_image, right_image):
        """Look for the tag in one rectified pair. True if the calibration changed."""
        if not self._resolution_ok:
            return False
        from xarm7_lib.tag_recovery import (
            ATTEMPTS_PER_FRAME, FRAME_QUORUM, LOOK_FRAMES,
        )

        now = time.monotonic()
        sighting = self._sight(left_image, right_image, now)
        lefts, rights, frames = self._batch
        frames += 1
        if sighting is not None:
            if self.seen_at is None or now - self.seen_at > TAG_QUIET:
                self._announce = True
            self.seen_at = now
            self.corners = sighting[0]
            self._expected = sighting
            lefts.append(sighting[0])
            rights.append(sighting[1])

        if len(lefts) < LOOK_FRAMES and frames < ATTEMPTS_PER_FRAME * LOOK_FRAMES:
            self._batch = (lefts, rights, frames)
            return False
        self._batch = ([], [], 0)
        if len(lefts) < np.ceil(FRAME_QUORUM * LOOK_FRAMES):
            return False

        # As `capture_tag` does it: medians, and the scatter turned into how far
        # those medians can be trusted.
        left, right = np.median(lefts, axis=0), np.median(rights, axis=0)
        scatter = 1.4826 * np.median(np.abs(np.asarray(lefts) - left))
        noise = float(1.25 * scatter / np.sqrt(len(lefts)))
        return self._judge(left, right, noise)

    def _sight(self, left_image, right_image, now):
        """The tag's corners in both eyes, `(left, right)`, or None."""
        from xarm7_lib.tag_recovery import MAX_ROW_ERROR

        left = self._find(left_image, self._expected[0])
        if left is None and now - self._full_at >= TAG_SEARCH_PERIOD:
            self._full_at = now
            left = self._find(left_image, None)
        if left is None:
            return None
        # The right eye sees it the same rows over, one disparity to the left,
        # and the disparity is about what it was.
        offset = np.mean(self._expected[0] - self._expected[1], axis=0)
        right = self._find(right_image, left - offset)
        if right is None:
            return None
        if np.max(np.abs(left[:, 1] - right[:, 1])) > MAX_ROW_ERROR:
            return None
        if np.any(left[:, 0] - right[:, 0] <= 0):
            return None
        return left, right

    def _find(self, image, near):
        """The tag in `image`, searched within `TAG_WINDOW` of `near` or everywhere."""
        from xarm7_lib.tag_recovery import find_tag

        if near is None:
            return find_tag(self._detector, image, self.reference.tag_id)
        height, width = image.shape[:2]
        x0, y0 = np.maximum(np.floor(near.min(axis=0) - TAG_WINDOW).astype(int), 0)
        x1, y1 = np.ceil(near.max(axis=0) + TAG_WINDOW).astype(int)
        x1, y1 = min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            return None
        found = find_tag(self._detector, image[y0:y1, x0:x1], self.reference.tag_id)
        return None if found is None else found + (x0, y0)

    def _judge(self, left, right, noise):
        """Decide what the calibration is now, from one batch of sightings."""
        from xarm7_lib.handeye import (
            MOVED_BEARING, MOVED_RANGE, MOVED_ROTATION, warn_moved,
            invert, rotation_angle, fit_camera_shift,
        )

        shift = fit_camera_shift(self.reference, left, right, noise)
        self.last_shift = shift
        announce, self._announce = self._announce, False
        tag = f"table tag {self.reference.tag_id}"

        # Is this a different camera pose from the one in use? Judged where the
        # tag is seen from each, with the thresholds that decide "moved" at all.
        centre = np.append(self.reference.points.mean(axis=0), 1.0)
        current = self.handeye.shift
        T_now = np.eye(4) if current is None else np.asarray(current.T_old_new)
        T_new = np.asarray(shift.T_old_new)
        seen_now = (invert(T_now) @ centre)[:3]
        seen_new = (invert(T_new) @ centre)[:3]
        cosine = seen_now @ seen_new / (np.linalg.norm(seen_now) * np.linalg.norm(seen_new))
        differs = (
            np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) > MOVED_BEARING
            or np.degrees(rotation_angle(T_now[:3, :3], T_new[:3, :3])) > MOVED_ROTATION
            or abs(np.linalg.norm(seen_now) - np.linalg.norm(seen_new)) > MOVED_RANGE
        )

        if differs:
            if shift.moved:
                self.handeye = self.filed.corrected(shift)
                warn_moved(self.filed, shift)
            else:
                self.handeye = self.filed
                _say(
                    f"checked against {tag}: the camera is back where it was "
                    "calibrated, so the correction is no longer used."
                )
            return True

        if announce:
            if current is None:
                _say(f"checked against {tag}: the camera is where it was calibrated.")
            else:
                _say(
                    f"checked against {tag}: the camera is still moved by "
                    f"{current.describe()}, and the correction for it is in use. "
                    "A TA should redo the calibration."
                )
        return False


class LightsaberDetector:
    """The light sticks, found by the camera and put where the arm can reach them.

    Owns the camera and the detection, and implements the `xarm7_lib`
    visualizer protocol, so

        with Robot() as robot, LightsaberDetector() as det:
            robot.register_visualizer(det)
            ...
            ends = det.query("red")

    is all it takes: both sticks appear beside the arm in meshcat, and the
    detection runs on its own thread, so a 100 ms grab never slows the picture.

    `query(color)` is the one a controller wants -- that stick's two ends in the
    frame `set_joint_targets` and the safety box are written in.

    It also watches for the bench's table tag in the frames it grabs, and keeps
    the hand-eye calibration in step with it (`TagWatch`): if the camera has been
    knocked, `query` and the drawing both move to a corrected transform and a
    warning is printed for a TA. At the exposure the sticks want, the tag is
    usually too dark to see, so the check happens whenever it is lit -- a phone
    flashlight on it for a few seconds is enough.

    Once it starts polling it also opens the same window `python -m
    libcapstone.light_stick` does: the left eye with each stick's outline, its
    ends and the numbers drawn on. The window runs on a thread of its own, so a
    slow redraw never holds up a grab. `q` (or closing it) closes the window and
    leaves the detector running; `window=False` never opens it.
    """

    # The frame the geometry below is written in. Base rather than camera, so
    # the picture goes through the same calibration as `query`, correction and
    # all; the library would otherwise load its own copy once and keep it.
    frame = "base"
    # Unpaced: the camera's own grab is the clock. There is no point asking for
    # frames faster than it can produce them.
    rate = None

    def __init__(self, resolution="2K", serial=None, device=None, fps=None,
                 colour=None, options=None, camera=None,
                 radius=0.015, max_age=0.5, name="blade",
                 window=True, display_scale=0.5):
        self._options = options if options is not None else Options()
        self._radius = float(radius)
        self._max_age = float(max_age)
        self._name = name

        # The live window. Opened by the first poll, so a detector that is
        # built and never run puts nothing on the screen.
        self._window = bool(window)
        self._display_scale = float(display_scale)
        self._display = None  # its thread, once started
        self._display_stop = threading.Event()
        self._frame = None  # (left, right, detections, fps) of the latest poll
        self._fps = 0.0
        self._polled_at = None

        self._lock = threading.Lock()
        self._latest = {}  # color -> (Detection, time.monotonic()) of the last good one
        self._last = {}  # color -> the last Detection either way, good or refused
        self._votes = {}  # color -> running say on which end is the hilt; the poll thread's own
        self._handeye = None  # as loaded at start, before the tag has a say
        self._handeye_error = None  # why there is none, once that is known
        self._watch = None  # the TagWatch, if the bench has a table tag
        self._setup_lock = threading.Lock()
        self._poll_lock = threading.Lock()  # one poll at a time, and none during close
        self._said_undrawn = False
        self._closed = False
        self._plane = None  # (geometry, material, transform) from `draw_plane`, or None

        # Built once, here, and never rebuilt: the library diffs the scene, so
        # geometry it has seen before is only moved.
        geometry = _meshcat_geometry()
        self._shaft = geometry.Cylinder(1.0, radiusBottom=1.0, radiusTop=1.0)
        self._materials = {
            color: geometry.MeshLambertMaterial(color=STICKS[color]["draw_color"]) for color in COLORS
        }

        if camera is not None:
            self._camera = camera
            self._owns_camera = False
        else:
            # Opened here, not on the first poll: a machine with no camera
            # should fail on the line somebody wrote, with a traceback they can
            # read, rather than three seconds later on a background thread.
            from xarm7_lib import ZedOpenCapture

            self._camera = ZedOpenCapture(
                resolution,
                device=device,
                serial=serial,
                fps=fps,
                controls=load_colour(colour),
            )
            self._owns_camera = True

    # ------------------------------------------------------------------
    # the visualizer protocol
    # ------------------------------------------------------------------

    @property
    def camera(self):
        """The camera being read, whether this opened it or was handed one."""
        return self._camera

    @property
    def serial(self):
        """Which camera this is, or None for a replay with no serial.

        The bench's hand-eye calibration is filed under it. Without it the
        library would go and ask the camera -- which means opening it, and this
        detector is streaming it, so the answer would come back empty. Also
        what the library reads off a visualizer drawing in camera coordinates.
        """
        return getattr(self._camera, "serial", None)

    def poll(self):
        """Grab a pair and look for both sticks. The slow half; nothing is held."""
        # Held for the whole poll, so `close` never lets the camera go mid-grab.
        with self._poll_lock:
            if self._closed:
                return
            self._poll()

    def _poll(self):
        if self._window and self._display is None:
            self._open_window()
        left, right = self._camera.read_rectified()
        watch = self.tag_watch
        if watch is not None:
            watch.feed(left, right)
        detections = detect_all(left, right, self._camera.calibration, self._options)
        # Anything that mutates a detection has to happen before it is
        # published below; after that it is shared with other threads.
        now = time.monotonic()
        with self._lock:
            previous = dict(self._latest)
        for color, detection in detections.items():
            if detection.ok:
                detections[color] = self._steady(color, detection, previous.get(color), now)
        if self._polled_at is not None:
            rate = 1.0 / max(now - self._polled_at, 1e-6)
            self._fps = 0.9 * self._fps + 0.1 * rate if self._fps else rate
        self._polled_at = now
        with self._lock:
            self._last.update(detections)
            for color, detection in detections.items():
                if detection.ok:
                    self._latest[color] = (detection, now)
            self._frame = (left, right, detections, self._fps)

    # ------------------------------------------------------------------
    # the live window
    # ------------------------------------------------------------------

    def _open_window(self):
        """Start the window's thread, or say why there will be no window."""
        reason = _no_window_reason()
        if reason is not None:
            self._window = False
            _say(f"not showing the camera window: {reason}. The detector is still running.")
            return
        self._display = threading.Thread(target=self._show, name="light-stick-window", daemon=True)
        self._display.start()

    def _show(self):
        """The window's own loop. Every OpenCV window call happens on this thread.

        Redraws only when a poll has produced a new frame, but pumps the
        window's events in between, so it stays responsive while a grab is slow.
        """
        scale = self._display_scale
        shown = None
        sized = False
        try:
            with _quiet_qt():
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            while not self._display_stop.is_set():
                with self._lock:
                    frame = self._frame
                if frame is not None and frame is not shown:
                    shown = frame
                    left, right, detections, fps = frame
                    if not sized:
                        height, width = left.shape[:2]
                        cv2.resizeWindow(WINDOW, int(scale * width), int(scale * height))
                        sized = True
                    overlay = draw_overlay(
                        left, detections, fps, tag=self._watch, debug=False
                    )
                    cv2.imshow(WINDOW, cv2.resize(overlay, None, fx=scale, fy=scale))
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27) or (sized and not _window_open(WINDOW)):
                    _say("the camera window is closed. The detector is still running.")
                    break
        except cv2.error as error:
            _say(f"the camera window stopped: {error}. The detector is still running.")
        finally:
            try:
                cv2.destroyWindow(WINDOW)
                cv2.waitKey(1)
            except cv2.error:
                pass

    def _steady(self, color, detection, previous, now):
        """Keep tip and hilt from swapping between frames.

        One frame's say on which end is the hilt is right most of the time,
        and "most of the time" would have the tip jump a quarter of a metre
        every few frames. So an end stays the end it was last frame, whichever
        it is nearest, and the frames' say accumulates in a running vote that
        only turns the stick round if it builds up against the current label.
        """
        vote = 0.0
        if previous is not None and now - previous[1] <= self._max_age:
            ends = previous[0].ends
            straight = np.linalg.norm(detection.ends - ends, axis=1).sum()
            crossed = np.linalg.norm(detection.ends[::-1] - ends, axis=1).sum()
            if crossed < straight:
                detection = detection.flipped()
            vote = VOTE_DECAY * self._votes.get(color, 0.0)
        vote += detection.hilt_evidence
        if vote < 0:
            detection, vote = detection.flipped(), -vote
        self._votes[color] = vote
        return detection

    def geometry(self):
        """What should be on screen now: each stick seen, and nothing else."""
        from xarm7_lib import HandEyeError

        shown = {}
        plane = self._plane
        if plane is not None:
            # Already in base metres, so it needs no calibration to be drawn.
            shown[f"{self._name}_plane"] = plane
        for color in COLORS:
            try:
                ends = self.query(color)
            except HandEyeError as error:
                # Camera metres drawn as base metres would be a confident
                # picture of the wrong place, so no stick is drawn instead.
                if not self._said_undrawn:
                    self._said_undrawn = True
                    _say(f"the sticks will not be drawn: {error}")
                return shown  # the plane alone, if there is one
            if ends is None:
                # Named nothing, so the library hides it. This is the whole
                # stale-blade policy: a blade still drawn four seconds after the
                # detector stopped seeing it is the worst outcome available
                # here, because it looks right.
                continue
            tip, hilt = ends
            shown[f"{self._name}_{color}"] = (
                self._shaft,
                self._materials[color],
                _segment_transform()(hilt, tip, self._radius),
            )
        return shown

    def draw_plane(self, plane, center=None, size=1.0, color="tab:blue", opacity=0.35):
        """Draw the plane `a x + b y + c z + d = 0` in the meshcat scene, or clear it.

        `plane` is `(a, b, c, d)` in the robot's base frame, metres -- the frame
        `query` answers in. It is drawn as a translucent `size` x `size` square
        centred on the point of the plane nearest `center` (a base-frame point;
        the base origin if None). `plane=None` removes it.

        `color` is anything matplotlib takes -- "red", "r", "tab:green",
        "#ff8800", an `(r, g, b)` tuple in 0..1 -- or a `0xRRGGBB` int.

        Shows up once the detector is registered with `robot.register_visualizer`,
        and stays until it is replaced or cleared. Call it as often as you like:
        only the square's pose is sent again, not its mesh.
        """
        if plane is None:
            self._plane = None
            return
        if not isinstance(color, (int, np.integer)):
            from matplotlib.colors import to_rgb

            color = int.from_bytes(bytes(round(255 * channel) for channel in to_rgb(color)), "big")
        color = int(color)
        a, b, c, d = (float(v) for v in np.asarray(plane, dtype=float).reshape(4))
        normal = np.array([a, b, c])
        norm = np.linalg.norm(normal)
        if not np.isfinite(norm) or norm < 1e-12 or not np.isfinite(d):
            raise ValueError(f"not a plane: {plane!r} (a, b, c must not all be zero)")
        normal, d = normal / norm, d / norm

        point = np.zeros(3) if center is None else np.asarray(center, dtype=float).reshape(3)
        point = point - (normal @ point + d) * normal  # its projection onto the plane

        # Any two unit vectors spanning the plane, with the normal as the box's z.
        helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, helper)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)

        transform = np.eye(4)
        transform[:3, :3] = np.column_stack((u, v, normal)) @ np.diag([size, size, 0.002])
        transform[:3, 3] = point

        # The mesh and material are kept between calls, so the library only
        # moves the square; a new one each call would be re-sent every time.
        previous = self._plane
        geometry = _meshcat_geometry()
        if previous is not None and getattr(previous[1], "_plane_look", None) == (color, opacity):
            box, material = previous[0], previous[1]
        else:
            box = previous[0] if previous is not None else geometry.Box([1.0, 1.0, 1.0])
            material = geometry.MeshLambertMaterial(
                color=color, opacity=float(opacity), transparent=opacity < 1.0
            )
            material._plane_look = (color, opacity)
        self._plane = (box, material, transform)

    def close(self):
        """Let the camera go and shut the window. Safe to call more than once.

        It may be called while another thread is still polling -- `with
        Robot() as robot, LightsaberDetector() as det:` closes the detector
        before the robot stops its visualizer thread -- so it waits out a poll
        in progress, and every poll after it does nothing.
        """
        with self._poll_lock:
            if self._closed:
                return
            self._closed = True
        self._display_stop.set()
        display = self._display
        if display is not None and display is not threading.current_thread():
            display.join(timeout=2.0)
        if self._owns_camera:
            self._camera.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    # ------------------------------------------------------------------
    # reading it
    # ------------------------------------------------------------------

    def detection(self, color):
        """The most recent `Detection` of this stick, good or refused, or None before the first."""
        check_color(color)
        with self._lock:
            return self._last.get(color)

    def endpoints(self, color, max_age=None):
        """This stick's two ends in camera metres, `(2, 3)` tip first, or None.

        None if it has not been seen yet, or if its last good look has gone
        stale. Stale is not a worse answer, it is a wrong one: the stick has
        moved.
        """
        check_color(color)
        limit = self._max_age if max_age is None else float(max_age)
        with self._lock:
            latest = self._latest.get(color)
        if latest is None:
            return None
        detection, when = latest
        if time.monotonic() - when > limit:
            return None
        return detection.ends

    def query(self, color, max_age=None):
        """This stick's two ends in the robot's base frame, `(2, 3)` metres, or None.

        `color` is "red" or "green". Tip first, as a best guess: see the module
        docstring for how far to trust which end is which.

        Raises `HandEyeError` if the bench has no calibration filed. Unlike the
        drawing, which just goes quiet, there is no safe degraded answer here:
        camera metres returned as base metres would be about a metre wrong and
        indistinguishable from a correct reading.
        """
        ends = self.endpoints(color, max_age=max_age)
        if ends is None:
            return None
        return self.handeye.points_to_base(ends)

    @property
    def handeye(self):
        """The hand-eye calibration `query` is using now, corrected if the camera has moved.

        Raises `HandEyeError` if the bench has no calibration filed.
        """
        from xarm7_lib import HandEyeError

        self._load_handeye()
        if self._handeye_error is not None:
            raise HandEyeError(self._handeye_error)
        watch = self._watch
        return self._handeye if watch is None else watch.handeye

    @property
    def tag_watch(self):
        """The `TagWatch` keeping the calibration honest, or None if there is no tag to watch."""
        self._load_handeye()
        return self._watch

    def _load_handeye(self):
        """Read the calibration once, as filed. The tag then gets its say frame by frame."""
        with self._setup_lock:
            if self._handeye is not None or self._handeye_error is not None:
                return
            from xarm7_lib import HandEyeError, load_handeye

            try:
                handeye = load_handeye(serial=self.serial)
            except HandEyeError as error:
                self._handeye_error = str(error)
                return
            if handeye.tag is not None:
                self._watch = TagWatch(handeye, getattr(self._camera, "resolution", None))
            self._handeye = handeye


def check_color(color):
    """Refuse a stick that does not exist, rather than quietly never seeing it."""
    if color not in COLORS:
        raise LightStickError(f"no {color!r} stick; there is {' and '.join(COLORS)}")


def _no_window_reason():
    """Why a window cannot be opened from a background thread here, or None if it can."""
    if sys.platform == "darwin":
        return "macOS only lets a program's main thread open windows"
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        # Checked here because a GUI toolkit with no display does not raise,
        # it aborts the whole process.
        return "there is no display to open it on (DISPLAY is not set)"
    return None


# What Qt 5 loads from a font directory. It reads that one directory, not below it.
QT_FONT_FILES = ("*.ttf", "*.ttc", "*.otf", "*.pfa", "*.pfb")


def _has_fonts(folder):
    folder = pathlib.Path(folder)
    return folder.is_dir() and any(next(folder.glob(p), None) for p in QT_FONT_FILES)


@contextlib.contextmanager
def _quiet_qt():
    """Around OpenCV's first window, which is when its Qt starts: head off Qt's warnings.

    Only the environment reaches Qt, and only as it starts.

    Fonts: opencv-python points `QT_QPA_FONTDIR` at a `cv2/qt/fonts` it does
    not ship, as soon as `cv2` is imported, and Qt complains about it every
    time it looks. They are only for the window's own toolbar -- the overlay
    text is drawn by OpenCV -- so any directory of system fonts will do, and a
    `QT_QPA_FONTDIR` that does hold fonts is left alone.

    Wayland: this Qt can only reach the screen through X11, and on a GNOME
    Wayland desktop it says it is ignoring Wayland whenever it sees
    `XDG_SESSION_TYPE=wayland`, even with `QT_QPA_PLATFORM` set. So while it
    starts it is told the session is X11, which is the choice it would have
    made anyway, and the variable is put back afterwards for everything else
    in the process.

    Exit: the detector's window starts Qt on a thread of its own, so Qt's
    objects belong to that thread, and when the process exits Qt tears them
    down from the main thread and warns `QObject::killTimer: Timers cannot be
    stopped from another thread` -- after everything has already finished, and
    harmless. That warning has no category of its own, so Qt's uncategorised
    warnings are turned off. Qt reads its logging rules once, as it starts, so
    this one stays set; rules somebody has set themselves are kept and win.
    """
    if not sys.platform.startswith("linux"):
        yield
        return
    rules = os.environ.get("QT_LOGGING_RULES", "")
    if "default.warning" not in rules:
        os.environ["QT_LOGGING_RULES"] = ";".join(r for r in ("default.warning=false", rules) if r)
    if not _has_fonts(os.environ.get("QT_QPA_FONTDIR", "")):
        for root in (pathlib.Path("/usr/share/fonts/truetype/dejavu"), pathlib.Path("/usr/share/fonts")):
            if not root.is_dir():
                continue
            found = next((d for d in (root, *sorted(root.rglob("*"))) if _has_fonts(d)), None)
            if found is not None:
                os.environ["QT_QPA_FONTDIR"] = str(found)
                break

    session = os.environ.get("XDG_SESSION_TYPE")
    disguise = (
        session == "wayland"
        and os.environ.get("DISPLAY")
        and "wayland" not in os.environ.get("QT_QPA_PLATFORM", "")
    )
    if disguise:
        os.environ["XDG_SESSION_TYPE"] = "x11"
    try:
        yield
    finally:
        if disguise:
            os.environ["XDG_SESSION_TYPE"] = session


def _window_open(name):
    """False once the window has been closed with its own button."""
    # Not every backend answers every property: -1 is "not supported" as well
    # as "no such window", so the visible flag is asked first and autosize only
    # where that has no answer. A window that is gone may also just raise.
    try:
        visible = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        if visible >= 0:
            return visible >= 1
        return cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) >= 0
    except cv2.error:
        return False


def _meshcat_geometry():
    import meshcat.geometry

    return meshcat.geometry


def _segment_transform():
    from xarm7_lib import segment_transform

    return segment_transform


def eye_offset(left, right, scale=EYE_CHECK_SCALE):
    """Global shift between the two eyes, as `(dx, dy, confidence)` in pixels."""

    def prepare(image):
        grey = cv2.cvtColor(cv2.resize(image, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        return np.float32(grey)

    (dx, dy), response = cv2.phaseCorrelate(prepare(left), prepare(right))
    return dx / scale, dy / scale, response


def draw_overlay(image, detections, fps=0.0, eyes=None, tag=None, debug=True):
    """The left eye with each stick, its ends and the numbers drawn on.

    `tag` is a `TagWatch`, whose status goes in the text and whose corners are
    outlined while it is in view. `debug` adds each end's label and position
    and a line per stick on how it was seen or why it was refused, and says
    when the eyes are not on the same rows (`eyes`, from `eye_offset`): what the
    standalone window and `--replay` want, and more than a student needs.
    """
    overlay = image.copy()
    lines = []
    if tag is not None:
        lines.append(tag.status())
        if tag.seen_at is not None and time.monotonic() - tag.seen_at < 0.5:
            corners = np.round(tag.corners).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [corners], True, (0, 255, 0), 2)
    for color, detection in detections.items():
        if detection.left is not None:
            edges = cv2.Canny(detection.left.mask * 255, 50, 150)
            overlay[edges > 0] = STICKS[color]["overlay_edge"]

        if detection.ends is not None:
            a, b = (tuple(int(v) for v in np.round(p)) for p in detection.ends_px)
            cv2.line(overlay, a, b, (255, 0, 255), 2)
            for point, position, error, label in zip(
                (a, b), detection.ends, detection.depth_error, ("tip", "hilt")
            ):
                cv2.circle(overlay, point, 18, (0, 255, 255), 3)
                if not debug:
                    continue
                cv2.putText(
                    overlay,
                    f"{label} ({position[0]:+.3f} {position[1]:+.3f} {position[2]:.3f}) +-{error:.3f}",
                    (point[0] + 24, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

        if not debug:
            continue
        if detection.ok:
            lines.append(
                f"{color}: {detection.distance:.3f} m away, {detection.length:.3f} m long, "
                f"depth +-{max(detection.depth_error):.3f} m, "
                f"residual {detection.residual:.2f} px"
            )
        else:
            lines.append(f"{color}: no detection: {detection.reason}")
    if fps:
        lines.append(f"{fps:.1f} fps")
    if debug and eyes is not None and abs(eyes[1]) > MAX_EYE_DY:
        lines.append(f"eyes {eyes[1]:+.1f} px apart vertically -- not a matched pair")
    for i, line in enumerate(lines):
        cv2.putText(overlay, line, (20, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return overlay


def summary(detection):
    """One line of plain numbers, for the log and for `--replay`."""
    if not detection.ok and detection.ends is None:
        return detection.reason
    a, b = detection.ends
    return (
        f"({a[0]:+.3f} {a[1]:+.3f} {a[2]:.3f}) -> ({b[0]:+.3f} {b[1]:+.3f} {b[2]:.3f}) "
        f"{detection.length:.3f} m long, {detection.distance:.3f} m away, "
        f"+-{max(detection.depth_error):.3f} m, residual {detection.residual:.2f} px, "
        f"{detection.rows_used} rows at {detection.row_sigma:.1f} px"
        + ("" if detection.ok else f" [REFUSED: {detection.reason}]")
    )


def as_dict(detection):
    return {
        "ok": detection.ok,
        "reason": detection.reason,
        "color": detection.color,
        "ends_m": None if detection.ends is None else detection.ends.tolist(),
        "ends_px": None if detection.ends_px is None else detection.ends_px.tolist(),
        "clipped": list(detection.clipped),
        "length_m": detection.length,
        "distance_m": detection.distance,
        "direction": None if detection.direction is None else detection.direction.tolist(),
        "disparity_px": list(detection.disparity),
        "depth_error_m": list(detection.depth_error),
        "diameter_m": list(detection.diameter),
        "hilt_evidence": detection.hilt_evidence,
        "residual_px": detection.residual,
        "rows_used": detection.rows_used,
        "row_sigma_px": detection.row_sigma,
        "timings_ms": {k: round(v, 2) for k, v in detection.timings.items()},
    }


def summaries(detections):
    """One line per stick, each saying which it is."""
    return "\n  ".join(f"{color}: {summary(d)}" for color, d in detections.items())


def replay(args):
    """Re-run the detector over frames saved earlier, and write overlays beside them."""
    from xarm7_lib.calibration import StereoCalibration

    options = Options.from_namespace(args)
    folder = pathlib.Path(args.replay)
    pairs = sorted(folder.glob("*_rect_left.png"))
    if not pairs:
        raise SystemExit(f"no *_rect_left.png under {folder}")

    if args.conf:
        calibration = StereoCalibration.from_file(
            args.conf, resolution=args.resolution, serial=args.serial
        )
    elif args.serial:
        calibration = StereoCalibration.from_serial(args.serial, resolution=args.resolution)
    else:
        raise SystemExit("--replay needs --conf SN<serial>.conf or --serial")
    log_calibration(calibration)

    out = pathlib.Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in pairs:
        name = path.name[: -len("_rect_left.png")]
        left = cv2.imread(str(path))
        right = cv2.imread(str(path.with_name(f"{name}_rect_right.png")))
        if left is None or right is None:
            LOG.warning("%s: could not read both eyes", name)
            continue
        detections = detect_all(left, right, calibration, options)
        results[name] = {color: as_dict(d) for color, d in detections.items()}
        print(f"{name}\n  {summaries(detections)}")
        cv2.imwrite(str(out / f"{name}_replay_overlay.png"), draw_overlay(left, detections))

    (out / "light_stick.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"\noverlays and light_stick.json in {out}")
    return 0


def live_tag_watch(camera):
    """A `TagWatch` for the live window, or None if this bench has no tag to watch."""
    from xarm7_lib import HandEyeError, load_handeye

    try:
        handeye = load_handeye(serial=camera.serial)
    except HandEyeError as error:
        LOG.info("not watching the table tag: %s", error)
        return None
    if handeye.tag is None:
        LOG.info("not watching the table tag: none is recorded with this calibration")
        return None
    LOG.info("watching table tag %d, recorded %s", handeye.tag.tag_id, handeye.tag.created)
    return TagWatch(handeye, camera.resolution)


def setup_logging(verbose, log_file):
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def log_calibration(calibration):
    focal, baseline = calibration.focal_length, calibration.baseline
    LOG.info(
        "calibration %s: focal %.1f px, baseline %.1f mm, %.1f px of disparity "
        "at 1 m and %.1f mm of depth per pixel there",
        calibration.serial,
        focal,
        1000 * baseline,
        focal * baseline,
        1000 / (focal * baseline),
    )


def load_colour(path=None):
    """Camera settings to stream with: the packaged bench profile by default.

    Worth applying before anything else. The camera starts on automatic white
    balance, which quietly makes one eye\'s mask bigger than the other\'s --
    and a mask that disagrees between the eyes is a length that disagrees with
    itself.
    """
    if path is None:
        path = pathlib.Path(__file__).resolve().parent / "data" / "camera_color_settings.json"
    try:
        settings = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError) as error:
        raise LightStickError(f"could not read the colour profile {path}: {error}")
    if not isinstance(settings, dict):
        raise LightStickError(f"{path} should hold a JSON object of camera settings")
    return settings


def save_frame(folder, index, raw, rectified, detections, overlay, args):
    out = pathlib.Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"frame{index:05d}"
    for name, image in (
        ("raw_left", raw[0]),
        ("raw_right", raw[1]),
        ("rect_left", rectified[0]),
        ("rect_right", rectified[1]),
        ("overlay", overlay),
    ):
        if image is not None:
            cv2.imwrite(str(out / f"{stem}_{name}.png"), image)
    for color, detection in detections.items():
        for eye, blade in (("left", detection.left), ("right", detection.right)):
            if blade is not None:
                cv2.imwrite(str(out / f"{stem}_mask_{color}_{eye}.png"), blade.mask * 255)
    results = {color: as_dict(d) for color, d in detections.items()}
    (out / f"{stem}.json").write_text(json.dumps(results, indent=1) + "\n")
    LOG.info("saved %s to %s", stem, out)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", metavar="DIR", help="re-run on *_rect_left/right.png saved there")
    parser.add_argument("--conf", metavar="FILE", help="SN<serial>.conf to calibrate --replay with")
    parser.add_argument("--resolution", default="2K", help="2K, FHD, HD or VGA (default %(default)s)")
    parser.add_argument("--serial", help="which camera, if several are attached")
    parser.add_argument("--device", help="open this node instead of searching")
    parser.add_argument("--fps", type=int, help="frame rate; lower it to cut the USB bandwidth")
    parser.add_argument("--exposure", type=int, help="exposure to start at; low is what this wants")
    parser.add_argument("--gain", type=int, help="gain to start at; low keeps the dark clean")
    parser.add_argument(
        "--colour",
        metavar="FILE",
        help="camera settings to apply before streaming; the camera starts on "
        "automatic white balance, which quietly makes one eye's mask bigger "
        "than the other's (default: the packaged bench profile)",
    )
    parser.add_argument(
        "--color-threshold",
        type=int,
        help="a stick pixel has this much of its colour beyond neutral, for every "
        "stick (default: each stick's own, from data/stick_colors.json)",
    )
    parser.add_argument("--bright-threshold", type=int, default=BRIGHT_THRESHOLD)
    parser.add_argument("--dilate", type=int, default=CORE_DILATE)
    parser.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
    parser.add_argument(
        "--max-depth-error",
        type=float,
        default=MAX_DEPTH_ERROR,
        metavar="METRES",
        help="refuse an answer whose ends are worse than this (default %(default)s)",
    )
    parser.add_argument(
        "--no-colour",
        action="store_true",
        help="stream with whatever the camera is already set to, applying no profile",
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="do not watch for the table tag (by default it is outlined and its "
        "check reported, and a flashlight on it shows whether the camera has moved)",
    )
    parser.add_argument("--display-scale", type=float, default=0.5, help="shrink the window")
    parser.add_argument("--no-window", action="store_true", help="headless, log only")
    parser.add_argument("--frames", type=int, help="stop after this many frames")
    parser.add_argument("--save-dir", default="/tmp/light_stick", help="where `s` and --replay write")
    parser.add_argument("--log-file", help="also write the log here")
    parser.add_argument("--log-every", type=int, default=30, help="summary every N frames")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every frame")
    return parser.parse_args(argv)


WINDOW = "light stick"


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.verbose, args.log_file)

    try:
        return _run(args)
    except LightStickError as error:
        raise SystemExit(str(error))


def _run(args):
    if args.replay:
        return replay(args)

    options = Options.from_namespace(args)

    from xarm7_lib import ZedOpenCapture

    camera = ZedOpenCapture(
        args.resolution,
        device=args.device,
        serial=args.serial,
        fps=args.fps,
        exposure=args.exposure,
        gain=args.gain,
        controls=None if args.no_colour else load_colour(args.colour),
    )
    exposure = camera.exposure
    try:
        calibration = camera.calibration
        log_calibration(calibration)
        tag = None if args.no_tag else live_tag_watch(camera)

        if not args.no_window:
            with _quiet_qt():
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                WINDOW,
                int(args.display_scale * camera.size[0]),
                int(args.display_scale * camera.size[1]),
            )

        index = 0
        paused = False
        eyes = None
        raw = rectified = overlay = None
        detections = {c: Detection(False, "nothing read yet", color=c) for c in COLORS}
        last = time.perf_counter()
        fps = 0.0

        while args.frames is None or index < args.frames:
            if not paused:
                started = time.perf_counter()
                raw = camera.read()
                grab = 1000 * (time.perf_counter() - started)

                started = time.perf_counter()
                rectified = calibration.rectify_images(*raw)
                rectify = 1000 * (time.perf_counter() - started)

                eyes = eye_offset(*rectified)
                if eyes[0] > 0:
                    LOG.warning("frame %d: the eyes are swapped (dx %+.1f px)", index, eyes[0])
                elif abs(eyes[1]) > MAX_EYE_DY:
                    LOG.warning(
                        "frame %d: eyes %+.1f px apart vertically after rectification",
                        index, eyes[1],
                    )

                if tag is not None:
                    tag.feed(*rectified)
                detections = detect_all(rectified[0], rectified[1], calibration, options)
                for detection in detections.values():
                    detection.timings = {"grab": grab, "rectify": rectify, **detection.timings}

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / (now - last)
                last = now

                LOG.debug("frame %d:\n  %s", index, summaries(detections))
                if args.log_every and index % args.log_every == 0:
                    LOG.info("frame %d, %.1f fps:\n  %s", index, fps, summaries(detections))
                index += 1

            if args.no_window:
                continue

            overlay = draw_overlay(rectified[0], detections, fps, eyes, tag)
            cv2.imshow(WINDOW, cv2.resize(overlay, None, fx=args.display_scale, fy=args.display_scale))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                LOG.info("%s", "paused" if paused else "running")
            elif key == ord("s"):
                save_frame(args.save_dir, index, raw, rectified, detections, overlay, args)
            elif key in (ord("["), ord("]")):
                step = -5 if key == ord("[") else 5
                exposure = camera.set_exposure(max(0, (exposure if exposure is not None else 50) + step))
                LOG.info("exposure -> %s", exposure)
    except KeyboardInterrupt:
        LOG.info("interrupted")
    finally:
        camera.close()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
