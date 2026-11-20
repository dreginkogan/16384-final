"""16-384 capstone: find the light sticks, and show them to the arm.

Copy this whole `libcapstone` directory next to your own code; there is
nothing to install. `import libcapstone` then finds it because it is in the
working directory, the same way your own modules are found.

`LightsaberDetector` is the whole of the public surface: it owns the camera,
runs the detection, draws both sticks into the arm's meshcat scene, and reports
where each one is in the frame the arm takes commands in.

    with Robot() as robot, LightsaberDetector() as det:
        robot.register_visualizer(det)
        ...
        ends = det.query("red")   # or "green"; (2, 3) metres, or None

Everything here is imported on demand, because the detector needs OpenCV and
the detector module is where it is imported — so `import libcapstone` costs
nothing on a machine that is only running the simulator.

What the arm itself can do is `xarm7_lib`, which *is* installed; this only
adds the camera and the stick.
"""

_LIGHT_STICK_EXPORTS = {
    "COLORS": "light_stick",
    "Detection": "light_stick",
    "LightStickError": "light_stick",
    "LightsaberDetector": "light_stick",
    "Options": "light_stick",
    "ReplayCamera": "light_stick",
    "detect": "light_stick",
    "detect_all": "light_stick",
}


def __getattr__(name):
    module = _LIGHT_STICK_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_LIGHT_STICK_EXPORTS))


__all__ = sorted(_LIGHT_STICK_EXPORTS)
