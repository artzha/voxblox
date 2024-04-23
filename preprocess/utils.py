import cv2
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField

import std_msgs

def pub_pc_to_rviz(pc, pc_pub, ts, point_type="x y z", frame_id="os_sensor", seq=0, publish=True):
    if not isinstance(ts, rospy.Time):
        ts = rospy.Time.from_sec(ts)

    def add_field(curr_bytes_np, next_pc, field_name, fields):
        """
        curr_bytes_np - expect Nxbytes array
        next_pc - expects Nx1 array
        datatype - uint32, uint16
        """
        field2dtype = {
            "x":    np.array([], dtype=np.float32),
            "y":    np.array([], dtype=np.float32),
            "z":    np.array([], dtype=np.float32),
            "i":    np.array([], dtype=np.float32),
            "t":    np.array([], dtype=np.uint32),
            "re":   np.array([], dtype=np.uint16),
            "ri":   np.array([], dtype=np.uint16),
            "am":   np.array([], dtype=np.uint16),
            "ra":   np.array([], dtype=np.uint32),
            "r":    np.array([], dtype=np.float32),
            "g":    np.array([], dtype=np.float32),
            "b":    np.array([], dtype=np.float32)
        }
        field2pftype = {
            "x": PointField.FLOAT32,  "y": PointField.FLOAT32,  "z": PointField.FLOAT32,
            "i": PointField.FLOAT32,  "t": PointField.UINT32,  "re": PointField.UINT16,  
            "ri": PointField.UINT16, "am": PointField.UINT16, "ra": PointField.UINT32,
            "r": PointField.FLOAT32, "g": PointField.FLOAT32, "b": PointField.UINT32
        }
        field2pfname = {
            "x": "x", "y": "y", "z": "z", 
            "i": "intensity", "t": "t", 
            "re": "reflectivity",
            "ri": "ring",
            "am": "ambient", 
            "ra": "range",
            "r": "r",
            "g": "g",
            "b": "b"
        }

        #1 Populate byte data
        dtypetemp = field2dtype[field_name]

        next_entry_count = next_pc.shape[-1]
        next_bytes = next_pc.astype(dtypetemp.dtype).tobytes()

        next_bytes_width = dtypetemp.itemsize * next_entry_count
        next_bytes_np = np.frombuffer(next_bytes, dtype=np.uint8).reshape(-1, next_bytes_width)

        all_bytes_np = np.hstack((curr_bytes_np, next_bytes_np))

        #2 Populate fields
        pfname  = field2pfname[field_name]
        pftype  = field2pftype[field_name]
        pfpos   = curr_bytes_np.shape[-1]
        fields.append(PointField(pfname, pfpos, pftype, 1))
        
        return all_bytes_np, fields

    #1 Populate pc2 fields
    pc = pc.reshape(-1, pc.shape[-1]) # Reshape pc to N x pc_fields
    all_bytes_np = np.empty((pc.shape[0], 0), dtype=np.uint8)
    all_fields_list = []
    field_names = point_type.split(" ")
    for field_idx, field_name in enumerate(field_names):
        next_field_col_np = pc[:, field_idx].reshape(-1, 1)
        all_bytes_np, all_fields_list = add_field(
            all_bytes_np, next_field_col_np, field_name, all_fields_list
        )

    #2 Make pc2 object
    pc_msg = PointCloud2()
    pc_msg.width        = 1
    pc_msg.height       = pc.shape[0]

    pc_msg.header            = std_msgs.msg.Header()
    pc_msg.header.stamp      = ts
    pc_msg.header.frame_id   = frame_id
    pc_msg.header.seq        = seq

    pc_msg.point_step = all_bytes_np.shape[-1]
    pc_msg.row_step     = pc_msg.width * pc_msg.point_step
    pc_msg.fields       = all_fields_list
    pc_msg.data         = all_bytes_np.tobytes()
    pc_msg.is_dense     = True

    if publish:
        pc_pub.publish(pc_msg)

    return pc_msg

def get_pointsinfov_mask(points):
    """
    Assumes camera coordinate system input points
    """
    norm_p = np.linalg.norm(points, axis=-1, keepdims=True)

    forward_vec = np.array([0, 0, 1]).reshape(3, 1)
    norm_f = np.linalg.norm(forward_vec)
    norm_p[norm_p==0] = 1e-6 # Prevent divide by zero error

    angles_vec  = np.arccos( np.dot(points, forward_vec) / (norm_f*norm_p) )

    in_fov_mask = np.abs(angles_vec[:,0]) <= 1.57 #0.785398

    return in_fov_mask

def project_3dto2d_points(pc_np, calib_dict, use_rectified=True):
    """
    Project 3D points from lidar space to 2D image space using camera calibration information.

    Parameters:
        pc_np : numpy.ndarray
            Point cloud as a 2D array with (x, y, z) coordinates in columns.
        calib_ext_file : str or numpy.ndarray
            File path or 4x4 homogeneous matrix representing the extrinsic calibration information.
        calib_intr_file : str
            File path to the camera calibration YAML file containing intrinsic matrix and distortion coefficients.

    Returns:
        numpy.ndarray
            2D image points (x, y) obtained by projecting 3D points onto the image plane.
        numpy.ndarray
            Boolean mask indicating valid points within the camera field of view (True) and points outside (False).

    Notes:
        The function first transforms the point cloud from WCS to ego lidar space (if wcs_pose is provided).
        It then projects the 3D points onto the image plane using camera calibration matrices.
        The function returns 2D image points and a boolean mask indicating valid points within the camera field of view.
    """
    assert isinstance(calib_dict, dict), "calib_ext_file must be a dict"
    
    ext_homo_mat = calib_dict["extr"]["lidar2cam"]

    np.set_printoptions(suppress=True)
    #Load projection, rectification, distortion camera matrices
    if use_rectified:
        T_lidar_to_rect = calib_dict["extr"]['P']
        pc_homo = np.hstack((pc_np, np.ones((pc_np.shape[0], 1))))
        pc_rect_cam = T_lidar_to_rect @ pc_homo.T
        
        image_points= pc_rect_cam / pc_rect_cam[-1, :]

        MAX_INT32 = np.iinfo(np.int32).max
        MIN_INT32 = np.iinfo(np.int32).min
        image_points = np.clip(image_points.T, MIN_INT32, MAX_INT32)
        image_points = image_points.astype(np.int32)
    else:
        raise NotImplementedError

    valid_points_mask = get_pointsinfov_mask(
        (ext_homo_mat[:3, :3]@pc_np[:, :3].T).T+ext_homo_mat[:3, 3])

    return image_points, valid_points_mask

def apply_rgb_cmap(img_path, bin_np, calib_dict):
    image_pts, pts_mask = project_3dto2d_points(bin_np, calib_dict)

    in_bounds = np.logical_and(
            np.logical_and(image_pts[:, 0]>=0, image_pts[:, 0]<1224),
            np.logical_and(image_pts[:, 1]>=0, image_pts[:, 1]<1024)
        )

    valid_point_mask = in_bounds & pts_mask
    valid_point_indices = np.where(valid_point_mask)[0]
    valid_points = image_pts[valid_point_mask, :]
    pt_color_map = np.array([(255, 255, 255)] * bin_np.shape[0], dtype=np.uint32)

    image_np = cv2.imread(img_path, cv2.IMREAD_COLOR)

    pt_color_map[valid_point_indices, :3] = image_np[valid_points[:, 1], valid_points[:, 0]]
    pt_color_map = np.stack((pt_color_map[:, 2], pt_color_map[:, 1], pt_color_map[:, 0]), axis=-1)
    return pt_color_map, valid_point_mask
