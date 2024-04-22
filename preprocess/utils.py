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