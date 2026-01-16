import numpy as np
import open3d as o3d

def get_intrinsics(metadata):
    
    # camera intrinsics
    K = np.array(metadata['camera']['K']) 
    
    # Blender coordinate system -> OpenCV coordinate system
    K = np.abs(K.reshape(3, 3)) 

    # viewport resolution scaling
    h, w = metadata['metadata']['resolution']
    K[0, :] *= w
    K[1, :] *= h

    return K.astype(np.float32)

def get_extrinsics(metadata):
    """Returns camera extrinsics E_w2c (world to camera) of shape (N, 4, 4)"""
    import pyquaternion as pyquat

    def _get_R(quaternion):
        return pyquat.Quaternion(quaternion).rotation_matrix

    # get t
    ts = np.array(metadata['camera']['positions'])

    # get R (T_wc i.e. camera to world)
    quaternions = np.array(metadata['camera']['quaternions'])

    # build rigid transformation [R|t] -> (4, 4)
    Rts = []
    for t, quat in zip(ts, quaternions):
        R = _get_R(quat) # [3, 3]
        Rt = np.concatenate([R, t[:, None]], axis=1)  # [3, 3] | [3, 1] -> [3, 4]
        Rt = np.vstack([Rt, np.array([0, 0, 0, 1])])  # [4, 4]
        Rts.append(Rt)
    Rts = np.array(Rts)


    # Blender coordinate system -> OpenCV coordinate system (invert the bases for Y and Z axes)
    # source = https://github.com/google-research/kubric/issues/331#issue-2451901401
    cv_from_gl_transform = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
    cv_from_gl_transform = cv_from_gl_transform[np.newaxis, :, :]  # (1, 4, 4)
    
    # camera-to-world matrix encoding a transformation from homogenous camera coordinates to homogenous world coordinates.
    # source = https://github.com/google-research/kubric/tree/main/challenges/movi
    E_c2w = np.matmul(Rts, cv_from_gl_transform)  # (N, 4, 4)
    E_w2c = np.linalg.inv(E_c2w)  # (N, 4, 4)

    return E_w2c.astype(np.float32)


def get_cam_params(metadata):
    K = get_intrinsics(metadata)
    E = get_extrinsics(metadata)
    return {
        'K': K,
        'E': E
    }

def pixel2cam(uv_z, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (uv_z[:, :, 0] - cx) * uv_z[:, :, 2] / fx
    y = (uv_z[:, :, 1] - cy) * uv_z[:, :, 2] / fy
    z = uv_z[:, :, 2]

    cam_coords = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    return cam_coords

def cam2world(cam_coords, E):
    E_c2w = np.linalg.inv(E)  # (4, 4)
    
    R = E_c2w[0:3, 0:3]  # (3, 3)
    t = E_c2w[0:3, 3]    # (3,)

    H, W, _ = cam_coords.shape
    cam_coords_flat = cam_coords.reshape(-1, 3).T  # (3, H*W)

    world_coords_flat = R @ cam_coords_flat + t[:, None]  # (3, H*W)
    world_coords = world_coords_flat.T.reshape(H, W, 3)  # (H, W, 3)

    return world_coords

def get_3D_world_coords(depth_map: np.ndarray, cam_params: dict[str, np.ndarray]) -> np.ndarray:
    
    K = cam_params['K']
    E = cam_params['E']
    
    # pixel coordinates
    u_grid, v_grid = np.meshgrid(
        np.arange(0, 512),  # width
        np.arange(0, 512),   # height
        indexing='xy'
    )

    # assign depth to pixel coordinates
    z = depth_map[:, :, 0]  # (H, W)
    uv_z = np.stack([u_grid, v_grid, z], axis=-1)  # (H, W, 3)

    # convert to camera coordinates
    cam_coords = pixel2cam(uv_z, K)  # (H, W, 3)

    # convert to world coordinates
    world_coords = cam2world(cam_coords, E)  # (H, W, 3)

    return world_coords

def compute_motion_map(cam_params0: dict[str, np.ndarray], 
                       cam_params1: dict[str, np.ndarray], 
                       depth0: np.ndarray, 
                       fflow0: np.ndarray, 
                       threshold: float=None) -> tuple[np.ndarray, np.ndarray]: 
    """Built for Kubric dataset"""

    H, W, _ = depth0.shape

    # camera parameters
    # T0 = T_ciw | T1 = T_ci+1w | T
    T_c0w, T_c1w = cam_params0['E'], cam_params1['E'] # note: c0 => c_t and c1 => c_t+1 
    K = cam_params0['K']
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # get inputs ready
    depth0 = depth0.reshape(-1)
    fflow0 = fflow0[...,:2][:, :, [1, 0]].reshape(-1, 2) # because Kubric gives flow in (dy, dx) format aka (delta_row, delta_col)

    # warping using camera pose

    # get pixel coordinates
    u0, v0 = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    p0 = np.stack([u0, v0], axis=-1).reshape(-1, 2) # pixel coordinates

    # convert to 3D and bring them to camera frame
    z = depth0
    x = (p0[:, 0] - cx) * z / fx
    y = (p0[:, 1] - cy) * z / fy
    X0 = np.stack([x, y, z], axis=-1)  #(N, 3)

    # if filtering required later in future
    valid = np.ones(H*W, dtype=bool)
    X0_valid = X0[valid]
    fflow0_valid = fflow0[valid]
    p0_valid = p0[valid]

    # get relative transform
    # transforms points in cam0 frame to cam1 frame
    E_rel = T_c1w @ np.linalg.inv(T_c0w)

    # warp points using pose
    X0_homo = np.hstack([X0_valid, np.ones((X0_valid.shape[0], 1))])  # (N, 4)
    X1_warped_pose_homo = (E_rel @ X0_homo.T).T
    X1_warped_pose = X1_warped_pose_homo[:, :3]

    # backproject to pixel coordinates
    X1_warped_pose_pixel = (K @ X1_warped_pose.T).T
    p1_pose = np.stack([X1_warped_pose_pixel[:,0] / X1_warped_pose_pixel[:,2], 
                        X1_warped_pose_pixel[:,1] / X1_warped_pose_pixel[:,2]], axis=1)

    # now calculate induced flow
    induced_flow = p1_pose - p0_valid
    
    # and dynamic flow (= optical flow - camera-induced flow)
    dynamic_flow = fflow0_valid - induced_flow

    # build dynamic map
    dynamic_flow_full = np.full((H*W, 2), np.nan)
    dynamic_flow_full[valid] = dynamic_flow
    dynamic_flow_full = dynamic_flow_full.reshape(H, W, 2)

    # and finally thresholding
    mag_dynamic_flow = np.linalg.norm(dynamic_flow_full, axis=-1)
    threshold = threshold if threshold is not None else np.percentile(mag_dynamic_flow, 95)
    dynamic_mask = mag_dynamic_flow > threshold
    dynamic_mask = dynamic_mask.astype(bool)

    return dynamic_mask, mag_dynamic_flow

def visualize_point_cloud(world_coords: np.ndarray, rgb: np.ndarray = None):
    H, W, _ = world_coords.shape
    points = world_coords.reshape(-1, 3)  # (H*W, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if rgb is not None:
        rgb = rgb.reshape(-1, 3)  # (H*W, 4)
        colors = rgb[:, :3].reshape(-1, 3) / 255.0  # (H*W, 3)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])