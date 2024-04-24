"""
This script converts the poses in coda (ts x y z qw qx qy qz) to kitti format 
(r1 r2 r3 t1 r4 r5 r6 t2 r7 r8 r9 t3) where r1 r2 r3 are the first row of the rotation matrix,

"""

import os
from os.path import join
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pickle
import quaternion  # Adds support for quaternions to numpy

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Converts coda poses to kitti format')
    parser.add_argument('--indir', type=str, default='/home/data/coda', help='Directory containing the coda poses')
    parser.add_argument('--outdir', type=str, 
                        # default='/media/arthur/ExtremePro/removert/coda/data',
                        default='/home/voxblox_ws/src/voxblox/map_builder/data',
                        help='Directory to save the kitti poses')
    parser.add_argument('--seq', type=str, default='all', help='Sequence number, or all to convert all sequences')
    return parser.parse_args()

def pose_coda2kitti(in_path, out_path):
    """
    Converts coda poses to kitti format and saves them in specific directory
    """
    if type(in_path) == str:
        poses = np.loadtxt(in_path, dtype=np.float64)
    elif type(in_path) == np.ndarray:
        poses = in_path
    else:
        raise ValueError('Invalid input type for poses')
    kitti_poses = []
    for i in range(poses.shape[0]):
        pose = poses[i]
        kitti_pose = np.zeros((3, 4))
        kitti_pose[:3, :3] = R.from_quat([pose[5], pose[6], pose[7], pose[4]]).as_matrix() # qw qx qy qz
        kitti_pose[:3, 3] = pose[1:4]

        kitti_poses.append(kitti_pose.flatten())
    kitti_poses = np.array(kitti_poses)
    np.savetxt(out_path, kitti_poses)
    print(f'Saved poses in {out_path}')

def convert_coda2kitti(indir, outdir, seq):
    """
    Converts poses to kitti format and saves them in specific directory
    """
    print(f'---- Converting coda poses to kitti format for sequence {seq} ----')
    in_path = os.path.join(indir, 'poses', 'dense_global', f'{seq}.txt')
    out_path = os.path.join(outdir, 'poses', f'{seq}.txt')
    if not os.path.exists(os.path.dirname(out_path)):
        out_pathdir = os.path.dirname(out_path)
        print(f'Creating directory {out_pathdir}')
        os.makedirs(out_pathdir)
    
    print(f'---- Loading poses from {in_path} and saving to {out_path} ----')
    pose_coda2kitti(in_path, out_path)

def load_poses(indir, seq_list):
    """
    Load poses for all sequences
    """
    output = {
        'global_infos': [],
        'poses': {}
    }
    for seq in seq_list:
        in_path = os.path.join(indir, 'poses', 'dense_global', f'{seq}.txt')
        output['poses'][seq] = np.loadtxt(in_path, dtype=np.float64)
        output['global_infos'].extend(
            [(int(seq), frame) for frame in range(output['poses'][seq].shape[0])]
        )
    
    return output

def quaternion_angular_difference(q1_array, q2_array):
    """
    Calculates the angular difference in radians between two quaternions.
    """
    # Ensure inputs are quaternions
    # q1 = quaternion.as_quat_array(q1)
    # q2 = quaternion.as_quat_array(q2)
    # Ensure the input is in quaternion array form
    q1 = np.array([np.quaternion(*q) for q in q1_array])
    q2 = np.array([np.quaternion(*q) for q in q2_array])
    
    # Compute the relative rotation quaternion array
    qr = q2 * np.conjugate(q1)
    
    # Normalize the quaternion array to ensure they are unit quaternions
    qr = quaternion.as_float_array(qr)
    qr /= np.linalg.norm(qr, axis=1)[:, np.newaxis]  # Normalize each quaternion
    
    # Calculate the angle of rotation (in radians) for each quaternion
    angles = 2 * np.arccos(np.clip(qr[:, 0], -1, 1))  # qr[:, 0] corresponds to the scalar 'w' component
    
    # Convert radians to degrees
    angles_degrees = np.degrees(angles)
    
    return angles_degrees

def filter_poses(pose_dict, angular_threshold=0.2, translation_threshold=0.08):
    """
    Filters poses in pose_dict toonly include poses that do not have large change in rotations with
    consecutive frmaes in the same sequence
    pose_dict:
        "global_infos": [(seq, frame), ...]
        "poses": {seq: np.array([[ts, x, y, z, qw, qx, qy, qz], ...]), ...}
    """
    global_infos = np.array(pose_dict['global_infos'])
    sequences = np.unique(global_infos[:, 0])

    # compute quaternion angle difference between consecutive frames ina  vecorized manner
    global_infos_mask = np.zeros(global_infos.shape[0], dtype=bool)
    for seq in sequences:
        seq_start_idx = np.where(global_infos[:, 0] == seq)[0][0]
        seq_end_idx = np.where(global_infos[:, 0] == seq)[0][-1]
        seq_poses = pose_dict['poses'][seq]

        # Compute quaternion angular difference between consecutive frames
        quaternions = seq_poses[:, 4:]
        quaternions = np.concatenate([quaternions[:, 1:], quaternions[:, :1]], axis=1)
        angular_diffs = quaternion_angular_difference(quaternions[:-1], quaternions[1:])

        xyzs = seq_poses[:, 1:4]
        diffs = np.linalg.norm(xyzs[:-1] - xyzs[1:], axis=1)

        # Filter out poses with large angular difference or small translation difference
        valid_poses = angular_diffs < angular_threshold
        valid_poses = np.logical_and(valid_poses, diffs > translation_threshold)
        valid_poses = np.concatenate([[True], valid_poses], axis=0) # Always include first pose
        pose_dict['poses'][seq] = seq_poses[valid_poses]
        global_infos_mask[seq_start_idx:seq_end_idx+1] = valid_poses

        # Plot heatmap of angular differences at each pose on 2d map plot
        fig, ax = plt.subplots()
        # normalize angular diffs for color mapping
        rgb_diffs = (angular_diffs - np.min(angular_diffs)) / (np.max(angular_diffs) - np.min(angular_diffs))
        x = pose_dict['poses'][seq][:, 1][1:]
        y = pose_dict['poses'][seq][:, 2][1:]
        rgb_diffs = rgb_diffs[valid_poses[1:]]

        scatter = ax.scatter(x, y, c=rgb_diffs, cmap='viridis')
        cbar = fig.colorbar(scatter)
        cbar.set_label('Normalized Angular Difference')
        plt.savefig("test.png")

    print(f'Filtered {np.sum(~global_infos_mask)} poses')
    # Filter global infos
    pose_dict['global_infos'] = global_infos[global_infos_mask]
    pose_dict['cluster_labels'] = pose_dict['cluster_labels'][global_infos_mask]

    return pose_dict


def cluster_poses(pose_dict):
    """
    Cluster poses using KMeans
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Automatically determine the number of clusters, stop when the silhouette score is maximized in search range

    poses = np.concatenate([pose_dict['poses'][seq] for seq in pose_dict['poses']], axis=0)
    xyz = poses[:, 1:4]
    scaler = StandardScaler()
    xyz = scaler.fit_transform(xyz)

    K_range = range(5, 15)
    silhouette_scores = []
    for K in K_range:
        kmeans = KMeans(n_clusters=K, random_state=0).fit(xyz)
        silhouette_scores.append(kmeans.inertia_)
    
    K = K_range[np.argmin(silhouette_scores)]
    print(f'Optimal number of clusters: {K}')
    kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto').fit(xyz)
    pose_dict['cluster_labels'] = kmeans.labels_
    pose_dict['cluster_centers'] = scaler.inverse_transform(kmeans.cluster_centers_)

    return pose_dict

def generate_unique_colors(n):
    """
    Generates n unique colors.

    Args:
    n (int): Number of unique colors to generate.

    Returns:
    list: A list of n unique RGB colors.
    """
    hues = np.linspace(0, 1, n+1)[:-1]  # exclude the last point, which is exactly one
    colors = [plt.cm.hsv(h) for h in hues]  # Use HSV colormap
    return colors

def visualize_pose_clusters(pose_dict):
    """
    Visualize the poses on 2d plot and color them by cluster

    Draw cluster center
    """
    cluster_centers = pose_dict['cluster_centers']
    num_clusters = len(cluster_centers)
    cluster_id_to_color = generate_unique_colors(num_clusters)
    fig, ax = plt.subplots()

    start_idx = 0
    for seq in pose_dict['poses']:
        poses = pose_dict['poses'][seq]
        end_idx = start_idx + poses.shape[0]
        cluster_labels = pose_dict['cluster_labels'][start_idx:end_idx]
        start_idx = end_idx
        cluster_colors = np.array(cluster_id_to_color)[cluster_labels]
        ax.scatter(poses[:, 1], poses[:, 2], c=cluster_colors)
    
    cluster_centers = pose_dict['cluster_centers']
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=cluster_id_to_color, marker='x', s=200)
    # Show legend with cluster idx for each color
    for i, cluster_center in enumerate(cluster_centers):
        ax.text(cluster_center[0], cluster_center[1], str(i), fontsize=12)
    plt.savefig(f'clusters{num_clusters}.png')
    fig.clear()

def prepare_pose_dict(pose_dict):
    """
    Group poses into new directories based on cluster labels
    
    """
    group_pose_dict = {cluster_idx: {
        "infos": np.zeros((0, 2), dtype=np.int64), # (seq, frame
        "poses": np.zeros((0, 8), dtype=np.float64)
    } for cluster_idx in np.unique(pose_dict['cluster_labels'])}

    start_idx = 0
    global_infos = np.array(pose_dict['global_infos'])
    for seq in pose_dict['poses']:
        seq_poses = pose_dict['poses'][seq]
        end_idx = start_idx + seq_poses.shape[0]
        seq_infos = global_infos[start_idx:end_idx]
        seq_labels = pose_dict['cluster_labels'][start_idx:end_idx]

        start_idx = end_idx

        # Save poses and infos for each cluster
        unique_seq_labels = np.unique(seq_labels)
        for cluster_id in unique_seq_labels:
            group_pose_dict[cluster_id]['infos'] = np.concatenate(
                [group_pose_dict[cluster_id]['infos'], seq_infos[seq_labels == cluster_id]], axis=0
            )
            group_pose_dict[cluster_id]['poses'] = np.concatenate(
                [group_pose_dict[cluster_id]['poses'], seq_poses[seq_labels == cluster_id]], axis=0
            )

    # Filter each cluster by max number of poses randomly
    # max_num_poses = 2000
    # np.random.seed(1337)
    # for cluster_id in group_pose_dict:
    #     num_pose_samples = min(max_num_poses, group_pose_dict[cluster_id]['poses'].shape[0])
    #     rand_ids = np.random.choice(group_pose_dict[cluster_id]['poses'].shape[0], num_pose_samples, replace=False)
    #     group_pose_dict[cluster_id]['infos'] = group_pose_dict[cluster_id]['infos'][rand_ids]
    #     group_pose_dict[cluster_id]['poses'] = group_pose_dict[cluster_id]['poses'][rand_ids]

    # Downsample consecutive poses in each cluster
    max_num_poses = 2000
    for cluster_id in group_pose_dict:
        cluster_num_poses = group_pose_dict[cluster_id]['poses'].shape[0]
        ds_factor = max(cluster_num_poses // max_num_poses, 1)
        group_pose_dict[cluster_id]['infos'] = group_pose_dict[cluster_id]['infos'][::ds_factor]
        group_pose_dict[cluster_id]['poses'] = group_pose_dict[cluster_id]['poses'][::ds_factor]

    return group_pose_dict

def create_symlink_files(group_pose_dict, indir, outdir):
    """
    Create symbolic link files for each cluster
    """
    for cluster_id in group_pose_dict:
        # cluster_dir = os.path.join(outdir, f'cluster_{cluster_id}')
        # if not os.path.exists(cluster_dir):
        #     print(f'Creating directory for {cluster_dir}')
        #     os.makedirs(cluster_dir)

        # Convert poses to kitti format
        pose_path = os.path.join(outdir, 'poses', f'{cluster_id}.txt')
        if not os.path.exists(os.path.dirname(pose_path)):
            print(f'Saving poses to {os.path.dirname(pose_path)}')
            os.makedirs(os.path.dirname(pose_path))
        pose_coda2kitti(
            group_pose_dict[cluster_id]['poses'],
            pose_path
        )

        # Symlink point cloud binary files
        for i, info in enumerate(group_pose_dict[cluster_id]['infos']):
            seq, frame = info
            filename = f'3d_comp_os1_{seq}_{frame}.bin'
            src_path = os.path.join(indir, '3d_comp', 'os1', f'{seq}', filename)
            dst_path = os.path.join(outdir, 'points', f'{cluster_id}', filename)
            if not os.path.exists(os.path.dirname(dst_path)):
                print(f'Saving symlink files to {os.path.dirname(dst_path)}')
                os.makedirs(os.path.dirname(dst_path))

            os.symlink(src_path, dst_path)
            print(f'Created symlink {dst_path}')

def dump_for_map_builder(outdir, pose_dict):
    """
    Saves pose_dict into correct format for map builder
    """
    if not os.path.exists(outdir):
        print(f'Creating output directory {outdir}')
        os.makedirs(outdir)

    # Save global infos
    global_infos = np.array(pose_dict['global_infos'], dtype=int)
    global_infos_path = join(outdir, 'global_infos.txt')
    np.savetxt(global_infos_path, global_infos, fmt='%d')

    # Save corresponding SE3 poses for global infos
    global_poses = np.concatenate([pose_dict['poses'][seq] for seq in pose_dict['poses']], axis=0)
    global_poses_quat   = global_poses[:, 4:][:, [1, 2, 3, 0]] # qw qx qy qz -> qx qy qz qw
    SO3_poses           = R.from_quat(global_poses_quat).as_matrix().reshape(-1, 9)
    SE3_poses           = np.concatenate([SO3_poses, global_poses[:, 1:4]], axis=1)
    SE3_poses_flat = SE3_poses.reshape(-1, 12)

    global_poses_path = join(outdir, 'global_poses.txt')
    np.savetxt(global_poses_path, SE3_poses_flat, fmt='%.6f')

    # Save corresponding cluster labels for global infos
    cluster_labels = pose_dict['cluster_labels']
    cluster_labels_path = join(outdir, 'cluster_labels.txt')
    np.savetxt(cluster_labels_path, cluster_labels, fmt='%d')
    print("Saved global infos, global poses, and cluster labels")

if __name__ == "__main__":
    args = parse_args()
    # assert args.seq not in ['8', '14', '15'], f'Global poses for {args.seq} are not available in CODa'

    # Cluster global poses across all sequences into K clusters
    if args.seq == 'all':
        seq_list = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21]
        if not os.path.exists('pose_dict.pkl'):
            pose_dict = load_poses(args.indir, seq_list)
            pose_dict = cluster_poses(pose_dict)
            
            # Save this pose_dict for future use using pickle
            with open('pose_dict.pkl', 'wb') as f:
                pickle.dump(pose_dict, f)
        else:
            with open('pose_dict.pkl', 'rb') as f:
                pose_dict = pickle.load(f)

        # Dump full pose dict for c++ map builder
        dump_for_map_builder(args.outdir, pose_dict)

        pose_dict = filter_poses(pose_dict)
        # Visualize the poses on 2d plot and color them by cluster
        visualize_pose_clusters(pose_dict)
        
        # Group poses by cluster label
        group_pose_dict = prepare_pose_dict(pose_dict)
    else:
        convert_coda2kitti(args.indir, args.outdir, args.seq)
