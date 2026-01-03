import numpy as np
import kubric as kb
from scipy.spatial.transform import Rotation
import pyquaternion as pyquat

def create_rectified_stereo_pair(
        position=[2, 2, 2],
        target=[0, 0, 0],
        up=[0, 1, 0], # Y-up
        front=[0, 0, -1], # -Z front
        baseline=0.54,
        focal_length=35.0,
):
    world_up = np.array([0, 0, 1], dtype=float) # +Z
    world_right = np.array([1, 0, 0], dtype=float) # +X

    # (up, front, right) -> camera coordinate system basis vectors
    # For right hand system, right = up x front
    up = np.array(up, dtype=float)
    up /= np.linalg.norm(up)

    front = np.array(front, dtype=float)
    front /= np.linalg.norm(front)

    right = np.cross(up, front)
    right /= np.linalg.norm(right)

    assert np.allclose(np.cross(right, up), front), "Basis vectors are not orthogonal! (right x up != front)"

    target = np.array(target, dtype=float)
    position = np.array(position, dtype=float)

    # desired coordinate system basis vectors 
    look_at_front = (target - position)
    look_at_front /= np.linalg.norm(look_at_front)

    look_at_right = np.cross(world_up, look_at_front)
    look_at_right /= np.linalg.norm(look_at_right)

    look_at_up = np.cross(look_at_front, look_at_right)
    look_at_up /= np.linalg.norm(look_at_up)

    rot_mat1 = np.stack([look_at_right, look_at_up, look_at_front])
    rot_mat2 = np.stack([right, up, front])
    
    quat = tuple(pyquat.Quaternion(matrix=(rot_mat1.T @ rot_mat2)))

    print(f"Look at right = {look_at_right}")
    left_pos = position - baseline * 0.5 * look_at_right
    right_pos = position + baseline * 0.5 * look_at_right

    left_cam = kb.PerspectiveCamera(
        name="left_camera",
        position=left_pos.tolist(),
        quaternion=quat,
        focal_length=focal_length,
    )

    right_cam = kb.PerspectiveCamera(
        name="right_camera",
        position=right_pos.tolist(),
        quaternion=quat,
        focal_length=focal_length,
    )

    print("Left camera position:", left_cam.position)
    print("Right camera position:", right_cam.position)

    print("Left camera quaternion:", left_cam.quaternion)
    print("Right camera quaternion:", right_cam.quaternion)

    return left_cam, right_cam
