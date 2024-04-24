#!/bin/bash

# Source the ROS environment
source /opt/ros/noetic/setup.bash  # Adjust the ROS version as necessary
source /home/voxblox_ws/devel/setup.bash  # Source your workspace setup file
export PYTHONPATH='/home/voxblox_ws/src/voxblox/preprocess':$PYTHONPATH

# Array of clusters to process
sequences=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)  # Adjust this array based on your sequence identifiers
sparsity=0.0  # Adjust the sparsity value as necessary
use_sparsity_compensation=false  # Set to true if you want to use sparsity compensation

# Loop through each sequence
for seq in "${sequences[@]}"; do
    echo "Processing sequence $seq"

    # Launch voxblox node
    roslaunch voxblox_ros coda_dataset.launch run_name:="$seq" use_sparsity_compensation:="$use_sparsity_compensation" sparsity:="$sparsity" &
    PID1=$!

    # Wait for the process to complete if necessary (optional)
    sleep 5  # Wait time in seconds, adjust according to your needs

    # Launch blocking cluster point cloud publisher
    python /home/voxblox_ws/src/voxblox/preprocess/coda_to_publisher.py --keyid "$seq" --indir /home/data/coda --dataset_type cluster
    
    # Make output mesh directory if dne
    mkdir -p "$(rospack find voxblox_ros)/mesh_results/sparse_$sparsity"

    # Use rosservice to save the sequence to a file
    result=$(rosservice call /voxblox_node/generate_mesh "{}")
    echo "Service completed with response: $result"

    check_file_path="$(rospack find voxblox_ros)/mesh_results/sparse_$sparsity/$seq.ply"

    # Check continuously if the file has been created
    while [ ! -f "$check_file_path" ]; do
        echo "Waiting for file to be created: $check_file_path"
        sleep 2  # Check every 2 seconds
    done
    echo "File created: $check_file_path"

    # Clean up voxblox process
    kill $PID1
    wait $PID1  # Wait for the process to clean up
done

echo "All sequences processed and saved successfully"
