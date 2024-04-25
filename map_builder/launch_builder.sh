#!/bin/bash

# Function to handle the SIGINT signal (Ctrl+C)
function handle_sigint {
    echo "Interrupt received, stopping..."
    exit 1  # Exit with a non-zero status to indicate that it was terminated by a signal
}

# Trap SIGINT (Ctrl+C) and call the handle_sigint function
trap 'handle_sigint' SIGINT

# Source the ROS environment
source /opt/ros/noetic/setup.bash  # Adjust the ROS version as necessary
source /home/voxblox_ws/devel/setup.bash  # Source your workspace setup file

# Define the range of cluster IDs
start_cluster_id=0
end_cluster_id=13  # Adjust this to set the number of clusters

# Path to the launch file
launch_file="$(rospack find map_builder)/launch/coda_classify.launch"

export OMP_NUM_THREADS=32  # Set the number of threads for OpenMP
# Loop through the range of cluster IDs
for cluster_id in $(seq $start_cluster_id $end_cluster_id)
do
  echo "Launching for cluster ID: $cluster_id"
  # Call roslaunch with overridden parameters
  roslaunch $launch_file cluster:="$cluster_id"
  echo "Done for cluster ID: $cluster_id"
done