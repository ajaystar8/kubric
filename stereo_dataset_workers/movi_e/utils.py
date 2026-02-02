import numpy as np
import kubric as kb
import pyquaternion as pyquat

def get_stereo_camera_positions(center_position, baseline, look_at_right):
    center_position = np.array(center_position, dtype=float)

    # just in case
    look_at_right = np.array(look_at_right, dtype=float)
    look_at_right /= np.linalg.norm(look_at_right)

    left_pos = center_position - baseline * 0.5 * look_at_right
    right_pos = center_position + baseline * 0.5 * look_at_right
    return left_pos, right_pos

def get_stereo_camera_pose(center_position, 
                            target=[0, 0, 0],
                            up=[0, 1, 0], # Y-up
                            front=[0, 0, -1], # -Z front
                            baseline=0.54):
    
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
    center_position = np.array(center_position, dtype=float)

    # desired coordinate system basis vectors 
    look_at_front = (target - center_position)
    look_at_front /= np.linalg.norm(look_at_front)

    look_at_right = np.cross(world_up, look_at_front)
    look_at_right /= np.linalg.norm(look_at_right)

    look_at_up = np.cross(look_at_front, look_at_right)
    look_at_up /= np.linalg.norm(look_at_up)

    rot_mat1 = np.stack([look_at_right, look_at_up, look_at_front])
    rot_mat2 = np.stack([right, up, front])
    
    quat = tuple(pyquat.Quaternion(matrix=(rot_mat1.T @ rot_mat2)))

    left_pos, right_pos = get_stereo_camera_positions(
        center_position=center_position,
        baseline=baseline,
        look_at_right=look_at_right,
    )

    assert np.allclose(np.linalg.norm(left_pos - right_pos), baseline), "Baseline distance does not match!"
    return {
        'quaternion': quat,
        'left_cam_position': left_pos,
        'right_cam_position': right_pos,
        'look_at_right': look_at_right,
    }

def create_rectified_stereo_pair(
        center_position=[2, 2, 2],
        target=[0, 0, 0],
        up=[0, 1, 0], # Y-up
        front=[0, 0, -1], # -Z front
        baseline=0.54,
        focal_length=35.0,
        sensor_width=32.0,
):
    
    # Compute the stereo camera poses
    pose_info = get_stereo_camera_pose(
        center_position=center_position,
        target=target,
        up=up,
        front=front,
        baseline=baseline,
    )
    quat = pose_info['quaternion']
    left_pos = pose_info['left_cam_position']
    right_pos = pose_info['right_cam_position']
    look_at_right = pose_info['look_at_right']

    # Create the camera objects
    left_cam = kb.PerspectiveCamera(
        name="left_camera",
        position=left_pos.tolist(),
        quaternion=quat,
        focal_length=focal_length,
        sensor_width=sensor_width,
    )

    right_cam = kb.PerspectiveCamera(
        name="right_camera",
        position=right_pos.tolist(),
        quaternion=quat,
        focal_length=focal_length,
        sensor_width=sensor_width,
    )

    # Sanity checks - ensure cameras are properly rectified
    assert np.allclose(left_cam.quaternion, right_cam.quaternion), "Left and right camera quaternions do not match!"
    assert np.allclose(np.linalg.norm(left_cam.position - right_cam.position), baseline), "Baseline distance does not match!"

    return {
        'left_camera': left_cam,
        'right_camera': right_cam,
        'quaternion': quat, # remains fixed for both cameras throughout the sequence
        'look_at_right': look_at_right, # required for computing the new camera positions during motion
    }

def sample_trajectory_velocity(rng, obj_position, speed_range, trajectory_type=None):
    """Sample velocity based on trajectory type, biased toward center."""
    
    types = ['horizontal', 'diagonal', 'rising', 'vertical']
    if trajectory_type is None or trajectory_type not in types:
        if trajectory_type is not None:
            print(f"Warning: Unknown trajectory type '{trajectory_type}'. Choosing randomly.")
        trajectory_type = rng.choice(types)
        print(f"Chose trajectory type: {trajectory_type}")
    
    speed = rng.uniform(speed_range[0], speed_range[1])
    pos = np.array(obj_position)
    
    # Direction toward center (origin)
    toward_center = -pos # vector from obj to origin
    toward_center = toward_center / (np.linalg.norm(toward_center) + 1e-8) # unit vector
    
    if trajectory_type == 'horizontal':
        # Mostly horizontal motion but biased toward center
        # Flatten the toward_center to x-y plane
        horizontal_dir = np.array([toward_center[0], toward_center[1], 0])
        horizontal_dir = horizontal_dir / (np.linalg.norm(horizontal_dir) + 1e-8)
        return tuple(speed * horizontal_dir + np.array([0, 0, rng.uniform(-1, 1)]))
    
    elif trajectory_type == 'diagonal':
        # 45-degree-ish entry toward center
        horizontal_dir = np.array([toward_center[0], toward_center[1], 0])
        horizontal_dir = horizontal_dir / (np.linalg.norm(horizontal_dir) + 1e-8)
        # Combine horizontal toward-center with downward z
        return tuple(speed * 0.7 * horizontal_dir + np.array([0, 0, -speed * 0.7]))
    
    elif trajectory_type == 'vertical':
        # Mostly falling/rising but drifting toward center
        drift = 0.2 * speed * toward_center[:2]  # slight x-y drift toward center
        return (drift[0], drift[1], rng.choice([-1, 1]) * speed)
    
    elif trajectory_type == 'rising':
        # Moving upward while heading toward center
        horizontal_dir = np.array([toward_center[0], toward_center[1], 0])
        horizontal_dir = horizontal_dir / (np.linalg.norm(horizontal_dir) + 1e-8)
        return tuple(speed * 0.5 * horizontal_dir + np.array([0, 0, speed * 0.8]))