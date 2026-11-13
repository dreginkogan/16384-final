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

robot = Robot(visualize=True, open_browser=True)

joints = robot.joint_values
vels = robot.joint_velocities
print(joints)
print()




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

def return_jacobian(kin_params, q):
    '''returns the xyz jacobian at a position 1'''
    # need to do this for pseudoinverse also
    T = transform(kin_params(q)) #7, 4, 4 -- transform for each of the 7 joints
    endeff_pos = T[-1, :3, 3] # the displacement column of the end effector

    J = np.zeros((3, 7))
    for i in range(7): # for each joint
        approach_i = T[i, :3, 2] # direction of motion, approach column of the transform
        origin_i = T[i, :3, 3] # transformation for the origin frame
        J[:, i] = np.cross(approach_i, endeff_pos - origin_i)
    return J

def pseudo_inverse(jacobian):
    """return left pseudoinverse"""
    return jacobian.T @ np.linalg.inv(jacobian @ jacobian.T)

def move_between_points(start_pos, end_pos, kinematic_parameters, steps=10, speed=0.2):
    pointDiff = end_pos - start_pos
    stepStartPos = start_pos

    for i in range(1, steps):
        stepStartPos = (pointDiff/steps)*i
        stepEndPos = (pointDiff/steps)*(i+1)
        diff = stepEndPos - stepStartPos

        jacobian = return_jacobian(kinematic_parameters, robot.joint_values)
        pseudo = pseudo_inverse(jacobian)
        joint_vels = pseudo@diff

        robot.set_joint_targets(robot.joint_values+joint_vels, speed, wait=True)


        
robot.show

# moves in straight lines in all 3 axes, as required
# move_between_points(np.array([0, 0, 0]), np.array([200, 0, 0]).T, kinematic_parameters, speed=0.15)
# move_between_points(np.array([200, 0, 0]), np.array([200, 200, 0]).T, kinematic_parameters, speed=0.15)
# move_between_points(np.array([200, 200, 0]), np.array([200, 200, 200]).T, kinematic_parameters, speed=0.15)
# move_between_points(np.array([200, 200, 0]), np.array([200, 200, 150]).T, kinematic_parameters, speed=0.15)


move_between_points(np.array([0, 0, 0]), np.array([100, 0, 100]).T, kinematic_parameters, speed=0.15)
robot.grip(1)
move_between_points(np.array([100, 0, 100]), np.array([100, 0, 0]).T, kinematic_parameters, speed=0.15)
robot.grip(0)
move_between_points(np.array([100, 0, 0]), np.array([150, 0, 150]).T, kinematic_parameters, speed=0.15)

# move_between_points(np.array([200, 200, 150]), np.array([200, 200, 50]).T, kinematic_parameters, speed=0.15)
# robot.grip(0)
# move_between_points(np.array([200, 200, 0]), np.array([200, 200, 200]).T, kinematic_parameters, speed=0.15)


robot.stop(wait=True)