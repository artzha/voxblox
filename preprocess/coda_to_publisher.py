"""
This script converts 3d point cloud bin files, corresponding RGB images, and known intrinsic and extrinsic calibrations to a ROSBag file.

"""

import os
from os.path import join
import argparse

import cv2
import numpy as np

from tqdm import tqdm
import multiprocessing as mp

import tf2_ros
import rospy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from utils import pub_pc_to_rviz, apply_rgb_cmap
from calibration import load_intrinsics, load_extrinsics

def parse_args():
    parser = argparse.ArgumentParser(description="Converts 3d point cloud bin files, corresponding RGB images, and known intrinsic and extrinsic calibrations to a ROSBag file.")
    parser.add_argument("--indir", required=True, type=str, help="Directory containing the dataset root.")
    parser.add_argument("--dataset_type", '-d', type=str, default="seq", help="Dataset type [seq, aggr]")
    parser.add_argument("--seq", type=int, default=0, help="Sequence number")
    parser.add_argument('--start_idx', '-sidx', type=int, default=0, help='Start frame')
    parser.add_argument('--end_idx', '-eidx', type=int, default=8213, help='End frame -1 means all frames')
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
    ts = pose_np[0]
    tf_msg = tf2_ros.TransformStamped()
    tf_msg.header.stamp = rospy.Time(ts)
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
        seq = args.seq
        start_idx = args.start_idx
        end_idx = args.end_idx

        # Load frame paths for publishing
        frame_infos = [(seq, i) for i in range(start_idx, end_idx)]
        data_infos = setup_frames(indir, frame_infos)
    else:
        raise NotImplementedError

    # Initialize ROS node
    rospy.init_node('coda_voxblox')

    publishers = [
        rospy.Publisher('/ouster/points', PointCloud2, queue_size=10),
        tf2_ros.TransformBroadcaster()
    ]

    #2 Publish frames sequentially tqdm visualize progress
    for i in tqdm(range(len(data_infos["pc"]))):
        publish_frame((
            publishers, 
            data_infos["pc"][i], 
            data_infos["pose"][i],
            data_infos["rgb"][i],
            data_infos["calibration"][i]
        ))    



if __name__=="__main__":
    args = parse_args()
    main(args)