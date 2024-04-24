#!/bin/bash

# Source the ROS environment
source /opt/ros/noetic/setup.bash  # Adjust the ROS version as necessary
source /home/voxblox_ws/devel/setup.bash  # Source your workspace setup file
export PYTHONPATH='/home/voxblox_ws/src/voxblox/preprocess':$PYTHONPATH

# Array of clusters to process
sequences=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)  # Adjust this array based on your sequence identifiers
sparsity=1.0

# Function to wait for file creation
wait_for_file() {
    local file_path=$1
    local seq=$2

    echo "Waiting for file to be created: $file_path"
    while [ ! -f "$file_path" ]; do
        sleep 2  # Check every 2 seconds
    done
    echo "File created: $file_path"
    # Clean up voxblox process
    local pid=$(pgrep -f "roslaunch.*$seq")
    if [ ! -z "$pid" ]; then
        kill $pid
        wait $pid  # Wait for the process to clean up
    fi
}

# Store all PIDs here
pids=()

# Launch all sequences in parallel
for seq in "${sequences[@]}"; do
    echo "Processing sequence $seq in parallel"

    # Make output mesh directory if doesn't exist
    mkdir -p "$(rospack find voxblox_ros)/mesh_results/sparse_$sparsity"

    # Launch voxblox node in the background
    export ROS_NAMESPACE="$seq"
    roslaunch voxblox_ros coda_dataset.launch run_name:="$seq" &
    PID1=$!
    pids+=($PID1) # Store PID

    # Launch the publisher in the background
    export ROS_NAMESPACE="$seq"
    python /home/voxblox_ws/src/voxblox/preprocess/coda_to_publisher.py --keyid "$seq" --indir /home/data/coda --dataset_type cluster &

    # Call rosservice to save the sequence to a file, also in the background
    (rosservice call /voxblox_node_$seq/generate_mesh "{}" && \
    wait_for_file "$(rospack find voxblox_ros)/mesh_results/sparse_$sparsity/$seq.ply" "$seq") &
done

# Now, wait for all background processes to finish
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All sequences processed and saved successfully"
