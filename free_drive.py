from xarm7_lib import Robot

robot = Robot(visualize=True, open_browser=True)
robot.free_drive()
input("press enter to end free drive and print joint values")
robot.end_free_drive()
print(robot.joint_values)