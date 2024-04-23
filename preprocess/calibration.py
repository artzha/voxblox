import os
import yaml
from os.path import join
import numpy as np

def load_intrinsics(indir, seq, camid):
    """
    Load the camera intrinsics from the calibration directory
    """
    calib_dir = join(indir, "calibrations")
    intrinsics = yaml.safe_load(open(join(calib_dir, str(seq), f'calib_{camid}_intrinsics.yaml'), 'r'))

    intrinsics_dict = {
        'K': np.array(intrinsics['camera_matrix']['data']).reshape(3,3),
        'R': np.array(intrinsics['rectification_matrix']['data']).reshape(3,3),
        'P': np.array(intrinsics['projection_matrix']['data']).reshape(
            intrinsics['projection_matrix']['rows'], intrinsics['projection_matrix']['cols']
        )
    }
    return intrinsics_dict

def load_extrinsics(indir, seq, camid):
    """
    Load the camera to LiDAR extrinsics from the calibration directory
    """
    calib_dir = join(indir, "calibrations")
    extrinsics = yaml.safe_load(open(join(calib_dir, str(seq), f'calib_os1_to_{camid}.yaml'), 'r'))

    extrinsics_dict = {
        'lidar2cam': np.array(extrinsics['extrinsic_matrix']['data']).reshape(
            extrinsics['extrinsic_matrix']['rows'], extrinsics['extrinsic_matrix']['cols']
        ),
        'P': np.array(extrinsics['projection_matrix']['data']).reshape(
            extrinsics['projection_matrix']['rows'], extrinsics['projection_matrix']['cols']
        )
    }
    return extrinsics_dict