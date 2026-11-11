from xarm7_lib import Robot
from libcapstone import LightsaberDetector

detector = LightsaberDetector()
with Robot() as robot:
    robot.register_visualizer(detector)

# Do things with the robot here. Free-drive as example.
robot.free_drive()
input('Enter to stop\n')
robot.end_free_drive()