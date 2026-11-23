from xarm7_lib import Robot
import numpy as np
np.set_printoptions(suppress=True)
from math import pi
from math import sin as s
from math import cos as c
from scipy.spatial.transform import Rotation, Slerp # so i can get angles, i am not doing all that math
import sys
from libcapstone import LightsaberDetector
from time import sleep
import matplotlib.pyplot as plt


# set vis to True when testing on computer, otherwise set IP for real arm
# with Robot(visualize=True, open_browser=True) as robot:
#     print("meow meow meow")

robot = Robot(visualize=True, open_browser=True)

joints = robot.joint_values
vels = robot.joint_velocities
# print(joints)
# print()

# from the capstone pdf
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
    '''returns the transform'''
    a, alpha, d, theta = params

    x_arr = np.array([[1, 0,        0,         a],
                      [0, c(alpha), -s(alpha), 0],
                      [0, s(alpha),  c(alpha), 0],
                      [0, 0,        0,         1]])

    z_arr = np.array([[c(theta), -s(theta), 0, 0],
                      [s(theta),  c(theta), 0, 0],
                      [0,         0,        1, d],
                      [0,         0,        0, 1]])

    return x_arr @ z_arr

def test_forward_kinematics(kin_params):
    '''goes through the test cases on the assignment pdf'''
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
    '''676767676767'''
    T = transform(kin_params(q))
    p_end = T[-1, :3, 3] # xyz position of the gripper

    J = np.zeros((6, 7))
    for i in range(7):
        z_axis = T[i, :3, 2] # z axis vector column
        origin = T[i, :3, 3] # xyz position
        J[:3, i] = np.cross(z_axis, p_end - origin) # linear velocity = z^0_{i-1} x (o_n - o_{i-1})
        J[3:, i] = z_axis # angular velocity = z^0_{i-1}
    return J

def get_position(q, kin_params):
    '''prints xyz location from transform'''
    return transform(kin_params(q))[-1, :3, 3]

def pseudo_inverse(jacobian):
    """return left pseudoinverse"""
    # return jacobian.T @ np.linalg.inv(jacobian @ jacobian.T)
    return np.linalg.pinv(jacobian)

def move_between_points(diff, rot_diff, kinematic_parameters, steps=5, speed=0.4, absRot = False):

    diff = np.asarray(diff, dtype=float)
    initT = transform(kinematic_parameters(robot.joint_values))[-1]

    startPos = initT[:3, 3]
    startRot = Rotation.from_matrix(initT[:3, :3])

    if absRot:
        rotGoal = rot_diff
    else:
        rotGoal = startRot * rot_diff

    slerp = Slerp([0, 1], Rotation.concatenate([startRot, rotGoal])) #not really sure what this is doing, but it is workingg

    for i in range(1, steps+1):
        stepEndPos = startPos + (diff/steps)*(i)
        # stepEndRot = startRot + (rot_diff/steps)*(i)

        # robot.set_joint_targets(robot.joint_values+joint_vels, speed, wait=True)


        q = inv_kin(stepEndPos, slerp(i/steps).as_matrix(), kinematic_parameters, robot.joint_values)
        robot.set_joint_targets(q, speed=speed, wait=False) if i<steps else robot.set_joint_targets(q, speed=speed, wait=True)


def inv_kin(p_target, r_target, kin_params, q0, iters = 10000, p_tolerance=0.02, r_tolerance=0.01, rot_weight=100):

    q = np.array(q0, dtype=float)
    r_target = Rotation.from_matrix(r_target) # turning into a Rotation object

    for k in range(iters):
        endeff_transform = transform(kin_params(q))[-1]


        p_error = p_target - endeff_transform[:3, 3]
        r_error = (r_target * Rotation.from_matrix(endeff_transform[:3, :3]).inv()).as_rotvec() # can't directly find difference by subtraction

        if np.linalg.norm(p_error) < p_tolerance and np.linalg.norm(r_error) < r_tolerance: # if within the tolerances, return, otherwise continue iterating
            return q

        J = return_jacobian(kin_params, q)
        error = np.concatenate([p_error, r_error*rot_weight]) # angles are in radians here, while dimensions are in mm -- rot weight makes rotations and transforms count more equally
        J[3:] *= rot_weight  

        h = J.T @ error # negative gradient
        Jh = J@h # Jacobnian @ negative gradient = 
        alpha = alpha = (h @ h) / (Jh @ Jh + 1e-12) # got from online 

        q = q+alpha*h # angles update equation from slides

    return q


arguments = sys.argv[1:] # got this online -- makes running commands and doing certain tasks easier

if len(arguments) != 1:
    print("ERROR: provide an argument for a specific task for the arm to do")
    sys.exit()

print("m1 for milestone 1")
print("m2invkintest")

argument = arguments[0]

measured_zero = [-0.04625704, -0.45930759,  0.09208055, -0.05188621,  0.09883064,  0.40833193, -0.053688  ] #arbitrary joints values for an xyz origin
robot.set_joint_targets(measured_zero, wait=True) # measured these irl

robot.show()

init_pos = get_position(robot.joint_values, kinematic_parameters)

print(f"initial position = {init_pos}")

input("press enter to run")


if argument == "m1":
    # milestone 1
    move_between_points(np.array([100, 0, 0]), Rotation.identity(), kinematic_parameters, speed=0.3)
    print(np.round(get_position(robot.joint_values, kinematic_parameters)-init_pos))

    move_between_points(np.array([0, 100, 0]), Rotation.identity(), kinematic_parameters, speed=0.3)
    print(np.round(get_position(robot.joint_values, kinematic_parameters)-init_pos))

    move_between_points(np.array([0, 0, 100]), Rotation.identity(), kinematic_parameters, speed=0.3)
    print(np.round(get_position(robot.joint_values, kinematic_parameters)-init_pos))
elif argument == "m2invkintest":
    # move_between_points(np.array([-100, 0, 0]), Rotation.identity(), kinematic_parameters, speed=0.3)


    R1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    move_between_points(np.array([500, 0, 0])-init_pos, Rotation.from_matrix(R1), kinematic_parameters, speed=0.5, absRot=True)
    print(np.round(get_position(robot.joint_values, kinematic_parameters)))

    R2 = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    move_between_points(np.array([100, 0, 300]), Rotation.from_matrix(R2), kinematic_parameters, speed=0.5, absRot=True)
    print(np.round(get_position(robot.joint_values, kinematic_parameters)))

    R3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    move_between_points(np.array([-200, 200, -50]), Rotation.from_matrix(R3), kinematic_parameters, speed=0.5, absRot=True)
    print(np.round(get_position(robot.joint_values, kinematic_parameters)))
elif argument=="m2":
    detector = LightsaberDetector()
    robot.register_visualizer(detector)

    R1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    move_between_points(np.array([300, 0, 0])-init_pos, Rotation.from_matrix(R1), kinematic_parameters, speed=0.5, absRot=True)
    
    ends1 = detector.query("red")
    print(ends1)

    R2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    move_between_points(np.array([250, -150, 200]), Rotation.from_matrix(R2), kinematic_parameters, speed=0.3)

    ends2 = detector.query("red")
    print(ends2)

    R3 = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    move_between_points(np.array([0, 0, -150]), Rotation.from_matrix(R3), kinematic_parameters, speed=0.3, absRot=True)

    ends3 = detector.query("red")
    print(ends3)
    
elif argument=="m2grasp":
    detector = LightsaberDetector()
    robot.register_visualizer(detector)

    R1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    move_between_points(np.array([300, 0, 0])-init_pos, Rotation.from_matrix(R1), kinematic_parameters, speed=0.5, absRot=True)
    robot.grip(1)
    input("put lightsaber block thing under, then press enter")
    print()
    move_between_points(np.array([0, 0, -80]), Rotation.identity(), kinematic_parameters, speed=0.3)
    robot.grip(0)
    move_between_points(np.array([0, 0, 80]), Rotation.identity(), kinematic_parameters, speed=0.3)

    ends1 = detector.query("red")
    print(ends1)

    R2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    move_between_points(np.array([250, -150, 200]), Rotation.from_matrix(R2), kinematic_parameters, speed=0.3)

    ends2 = detector.query("red")
    print(ends2)

    R3 = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    move_between_points(np.array([0, 0, -150]), Rotation.from_matrix(R3), kinematic_parameters, speed=0.3, absRot=True)

    ends3 = detector.query("red")
    print(ends3)
elif argument=="m2plane":
    detector = LightsaberDetector()
    robot.register_visualizer(detector)

    R1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    move_between_points(np.array([300, 0, 0])-init_pos, Rotation.from_matrix(R1), kinematic_parameters, speed=0.5, absRot=True)

    points = np.zeros((100, 2, 3))
    # print(points[1])
    for i in range(1, 101):
        reading = detector.query("red")
        points[i-1] = reading
        sleep(2/100)

    # vstack both base and tip points
    points = np.vstack((points[:, 0, :], points[:, 1, :]))

    # need to remove nan values actually

    # find centroid of all the points
    com = np.sum(points, axis=0)/len(points)

    # move com to origin
    q = points-com

    # find 3x3 matrix
    Q = np.dot(q.T, q)

    # find eigenvalues and eigenvectors
    la, vectors = np.linalg.eig(Q)

    # find eigenvector of smallest eigenvalue
    plane_v = vectors.T[np.argmin(la)]


    # how to find direction of least variance for the normal???
    plt.figure()
    ax = plt.subplot(111, projection='3d')

    # base_x = points[:, 0, 0][~np.isnan(points[:, 0, 0])]
    # base_y = points[:, 0, 1][~np.isnan(points[:, 0, 1])]
    # base_z = points[:, 0, 2][~np.isnan(points[:, 0, 2])]

    # tip_x = points[:, 1, 0][~np.isnan(points[:, 1, 0])]
    # tip_y = points[:, 1, 1][~np.isnan(points[:, 1, 1])]
    # tip_z = points[:, 1, 2][~np.isnan(points[:, 1, 2])]

    # xs = np.append(base_x, tip_x)
    # ys = np.append(base_y, tip_y)
    # zs = np.append(base_z, tip_z)

    # need to fit them



    print("yay")

print("complete")

# rot_target = Rotation.from_euler('xz', [90, 90], degrees=True)
# move_between_points(np.array([0, 0, 0]), rot_target, kinematic_parameters, speed=0.15)
# print(np.round(get_position(robot.joint_values, kinematic_parameters)-init_pos))

# move_between_points(np.array([0, 0, 0, 0, 0, 1]), np.array([0, 100, 0, 0, 0, 1]), kinematic_parameters, speed=0.15)
# print(np.round(get_position(robot.joint_values, kinematic_parameters)-init_pos))

# move_between_points(np.array([100, 100, 0, 0, 0, 1]), np.array([100, 100, 100, 0, 0, 1]), kinematic_parameters, speed=0.15)
# print(np.round(get_position(robot.joint_values, kinematic_parameters)-init_pos))

# print("after all the moving")
# move_between_points(np.array([100, 100, 100, 0, 0, 1]), np.array([100, 0, 0, 0, 0, 1]), kinematic_parameters, speed=0.15)

# robot.free_drive()
# input("press enter after the gripper is 10cm above the lightsaber")
# robot.end_free_drive()

# robot.grip(1)

# move_between_points(np.array([0, 0, 0]), np.array([0, 0, -100]).T, kinematic_parameters, speed=0.15)
# robot.grip(0)
# move_between_points(np.array([0, 0, 0]), np.array([0, 0, 200]).T, kinematic_parameters, speed=0.15)



# robot.stop(wait=True)


# q_now = np.array(robot.joint_values)
# T_now = transform(kinematic_parameters(q_now))[-1]

# p_goal = T_now[:3, 3] + np.array([200, 0, 0])
# R_goal = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()

# q_sol = inv_kin(p_goal, R_goal, kinematic_parameters, q_now)
# robot.set_joint_targets(q_sol, 0.15, wait=True)


# q_now = np.array(robot.joint_values)
# T_now = transform(kinematic_parameters(q_now))[-1]

# p_goal = T_now[:3, 3] + np.array([200, 200, 0])
# R_goal = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()

# q_sol = inv_kin(p_goal, R_goal, kinematic_parameters, q_now)
# robot.set_joint_targets(q_sol, 0.15, wait=True)