"""
This script converts 3d point cloud bin files, corresponding RGB images, and known intrinsic and extrinsic calibrations to a ROSBag file.

"""

import os
from os.path import join
import argparse

import cv2
import numpy as np
import pickle

from tqdm import tqdm
import multiprocessing as mp

import tf2_ros
import rospy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from utils import pub_pc_to_rviz, apply_rgb_cmap
from calibration import load_intrinsics, load_extrinsics
from coda_to_kitti import prepare_pose_dict, filter_poses

def parse_args():
    parser = argparse.ArgumentParser(description="Converts 3d point cloud bin files, corresponding RGB images, and known intrinsic and extrinsic calibrations to a ROSBag file.")
    parser.add_argument("--indir", required=True, type=str, help="Directory containing the dataset root.")
    parser.add_argument("--dataset_type", '-d', type=str, default="cluster", help="Dataset type [seq, cluster]")
    parser.add_argument("--keyid", type=int, default=0, help="Sequence number or cluster id")
    parser.add_argument("--start_idx", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_idx", type=int, default=-1, help="End frame index")
    return parser.parse_args()

def setup_frames(indir, infos):
    """
    Expects infos to contain (seq, frame) pairs that you would like to load paths for
    """

    pc_dir = join(indir, "3d_comp", "os1")
    rgb_dir = join(indir, "2d_rect", "cam0")
    pose_dir = join(indir, "poses", "dense_global")

    output = {"pc": [], "pose": [], "rgb": [], "calibration": []}
    pose_dict = {}
    calib_dict = {}
    for seq, frame in infos:
        pc_path = join(pc_dir, str(seq), f'3d_comp_os1_{seq}_{frame}.bin')
        rgb_path = join(rgb_dir, str(seq), f'2d_rect_cam0_{seq}_{frame}.png')
        if seq not in calib_dict:
            calib_dict[seq] = {}
            calib_dict[seq]["intr"] = load_intrinsics(indir, seq, "cam0")
            calib_dict[seq]["extr"] = load_extrinsics(indir, seq, "cam0")

        pose_path = join(pose_dir, f'{seq}.txt')
        if seq not in pose_dict:
            pose_dict[seq] = np.loadtxt(pose_path, dtype=np.float64)
        pose_np = pose_dict[seq][frame]

        output["pc"].append(pc_path)
        output["pose"].append(pose_np)
        output["rgb"].append(rgb_path)
        output["calibration"].append(calib_dict[seq])

    return output

def publish_frame(inputs):
    pub_list, pc_path, pose_np, rgb_path, calibration = inputs

    # Load pose transform
    ts = rospy.Time.now()   #pose_np[0]
    tf_msg = tf2_ros.TransformStamped()
    tf_msg.header.stamp = ts
    tf_msg.header.frame_id = "world"
    tf_msg.child_frame_id = "os_sensor"
    tf_msg.transform.translation.x = pose_np[1]
    tf_msg.transform.translation.y = pose_np[2]
    tf_msg.transform.translation.z = pose_np[3]
    tf_msg.transform.rotation.x = pose_np[5]
    tf_msg.transform.rotation.y = pose_np[6]
    tf_msg.transform.rotation.z = pose_np[7]
    tf_msg.transform.rotation.w = pose_np[4]

    # Load point cloud
    pc_xyz = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    pc_xyz = pc_xyz[:, :3] #Only take xyz

    if False:
        # Apply rgb colormap to point cloud
        pc_rgb, pc_mask = apply_rgb_cmap(rgb_path, pc_xyz, calibration)
        pc_xyz = pc_xyz[pc_mask, :]
        pc_rgb = pc_rgb[pc_mask, :].astype(np.float32) / 255.0

        point_type="x y z r g b"
        pc_xyz = np.hstack((pc_xyz, pc_rgb))
    else:
        point_type="x y z"

    # Publish
    pub_pc_to_rviz(pc_xyz, pub_list[0], ts, point_type, frame_id="os_sensor", publish=True)
    pub_list[1].sendTransform(tf_msg)

def main(args):
    indir = args.indir
    dataset_type = args.dataset_type

    if dataset_type == "seq":
        seq = args.keyid
        start_idx = args.start_idx
        end_idx = args.end_idx
        if end_idx == -1:
            # compute number of frames in directory by using poses
            pose_path = join(indir, "poses", "dense_global", f"{seq}.txt")
            pose_dict = np.loadtxt(pose_path, dtype=np.float64)
            end_idx = pose_dict.shape[0]

        # Load frame paths for publishing
        frame_infos = np.array([[seq, i] for i in range(start_idx, end_idx)])
        data_infos = setup_frames(indir, frame_infos)
    elif dataset_type == "cluster":
        seq = int(args.keyid)

        processed_pkl_path = join(indir, "/home/voxblox_ws/src/voxblox/preprocess/filtered_pose_dict.pkl")
        if os.path.exists(processed_pkl_path):
            with open(processed_pkl_path, "rb") as f:
                pose_dict = pickle.load(f)
        else:
            pkl_path = join(indir, "/home/voxblox_ws/src/voxblox/preprocess/pose_dict.pkl")
            with open(pkl_path, "rb") as f:
                pose_dict = pickle.load(f)

            pose_dict = filter_poses(pose_dict)
            with open(processed_pkl_path, "wb") as f:
                pickle.dump(pose_dict, f)

        group_pose_dict = prepare_pose_dict(pose_dict)
        frame_infos = group_pose_dict[seq]["infos"]
    else:
        raise NotImplementedError

    data_infos = setup_frames(indir, frame_infos)

    # Initialize ROS node
    rospy.init_node(f'coda_voxblox')
    rate = rospy.Rate(2)

    publishers = [
        rospy.Publisher(f'/ouster/points', PointCloud2, queue_size=10),
        tf2_ros.TransformBroadcaster()
    ]

    #2 Publish frames sequentially tqdm visualize progress
    num_frames = len(data_infos["pc"])
    # num_frames = min(1000, len(data_infos["pc"]))
    for i in tqdm(range(num_frames)):
        publish_frame((
            publishers, 
            data_infos["pc"][i], 
            data_infos["pose"][i],
            data_infos["rgb"][i],
            data_infos["calibration"][i]
        ))    
        rate.sleep()

if __name__=="__main__":
    args = parse_args()
    main(args)