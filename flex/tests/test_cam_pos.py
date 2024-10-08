from scipy.spatial.transform import Rotation as R
import numpy as np


cam_pos = np.array([-0.77542536, -0.02539806,  0.30146208])

cam_pos_normalised = cam_pos / np.linalg.norm(cam_pos)

rotation = R.align_vectors(cam_pos_normalised, np.array([1,0,0]))[0]
print("cam euler according to pos", rotation.as_euler("xyz", degrees=True))
print("cam quat according to pos", rotation.as_quat())

# test for getting camera position from camera quaternion
camera_euler = np.array([ 170.01676212 , -21.23421399, -178.12402112])
camera_euler = np.array([ 0.0, 22.0, 3.0])
# corresponds to [0,22,3]
camera_quat = np.array([-0.005696068282031459, 0.19181093598117868, 0.02913152799094475, 0.9809829120433564])
camera_rotation_test = R.from_quat(camera_quat)
# camera_rotation_test = R.from_euler("xyz", camera_euler, degrees=True)
# print out the euler angles
print("euler", camera_rotation_test.as_euler("xyz", degrees=True))
camera_forward = np.array([1, 0, 0])
world_forward = camera_rotation_test.apply(camera_forward)


print("rotation_1_normalised: ", cam_pos_normalised)
print("rotation_2_normalised: ", world_forward)
