from xarm7_lib import Robot
import numpy as np
np.set_printoptions(suppress=True)
from math import pi
from math import sin as s
from math import cos as c

# from libcapstone import LightsaberDetector

# detector = LightsaberDetector()

# set vis to True when testing on computer, otherwise set IP for real arm
# with Robot(visualize=True, open_browser=True) as robot:
#     # robot.register_visualizer(detector)
#     print("meow meow meow")

robot = Robot(visualize=True, open_browser=False)

joints = robot.joint_values
vels = robot.joint_velocities
print(joints)
print()

initial_pos = np.zeros(7)

# robot.show()
robot.set_joint_targets(initial_pos, speed=None, wait=True)
robot.stop(wait=True)



kinematic_parameters = lambda q:  np.array([[0, 0, 267, q[0]],
                                            [0, -pi/2, 0, q[1]],
                                            [0, pi/2, 293, q[2]],
                                            [52.5, pi/2, 0, q[3]],
                                            [77.5, pi/2, 342.5, q[4]],
                                            [0, pi/2, 0, q[5]],
                                            [76, -pi/2, 269, q[6]]])


def transform(kin_params):
    '''returns transformation matrices for each joint'''
    transforms = np.zeros((kin_params.shape[0], 4, 4))

    transforms[0, :, :] = denavit_hartenberg(kin_params[0, :])

    for i in range(1, kin_params.shape[0]):
        transforms[i, :, :] = transforms[i-1, :, :] @ denavit_hartenberg(kin_params[i, :])

    return transforms

def denavit_hartenberg(params):
    '''returns a transformation'''

    # just to make it less messy idk
    dx = params[0]
    a = params[1]
    dz = params[2]
    t = params[3]

    x_arr = np.array([[1, 0, 0, dx],
                      [0, c(a), -s(a), 0],
                      [0, s(a), c(a), 0],
                      [0, 0, 0, 1]])

    z_arr = np.array([[c(t), -s(t), 0, 0],
                      [s(t), c(t), 0, 0],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])

    return x_arr @ z_arr

def test_forward_kinematics(kin_params):
    '''goes through the test cases on the assignment'''
    test_cases = np.array([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0.5, 0, 1.2, 0, 0.7, 0],
                           [0.5, -0.5, 0.3, 1.0, -0.4, 0.8, 0.2],
                           [-0.8, 0.6, 1.0, 1.5, 0.5, 1.0, -1.2]])
    
    print()
    print("==== TESTING GRIPPER TRANSFORMS ====")
    for i in range(test_cases.shape[0]):
        print()
        print(f"= CONFIGURATION {i} =")
        print()
        print("Joint Angles:")
        print(test_cases[i])
        print()
        print("Expected Gripper Transform:")
        trans = transform(kin_params(test_cases[i]))[-1]
        print(trans.round(2))

    print()



# all works, my fk implementation is good
test_forward_kinematics(kinematic_parameters)