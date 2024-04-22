#!/bin/bash
cd /home/voxblox_ws
catkin init
catkin config --extend /opt/ros/noetic
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin config --merge-devel
# Setup voxblox
cd /home/voxblox_ws/src
wstool init . ./voxblox/voxblox_ssh.rosinstall
wstool update
cd /home/voxblox_ws/src
catkin build voxblox_ros