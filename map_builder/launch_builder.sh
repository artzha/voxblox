#!/bin/bash

# Source the ROS environment
source /opt/ros/noetic/setup.bash  # Adjust the ROS version as necessary
source /home/voxblox_ws/devel/setup.bash  # Source your workspace setup file

# Define the range of cluster IDs
start_cluster_id=0
end_cluster_id=13  # Adjust this to set the number of clusters

# Path to the launch file
launch_file="$(rospack find voxblox_ros)/map_builder/launch/coda_classify.launch"

# Loop through the range of cluster IDs
for cluster_id in $(seq $start_cluster_id $end_cluster_id)
do
  echo "Launching for cluster ID: $cluster_id"
  # Call roslaunch with overridden parameters
  roslaunch $launch_file cluster:=$cluster_id
done