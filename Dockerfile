# Use ROS Noetic official image
FROM ros:noetic-ros-base-focal

# Set the working directory
WORKDIR /workspace

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libomp-dev \
    libeigen3-dev \
    libopencv-dev \
    libpcl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install voxblox dependencies
RUN apt-get update && apt-get install -y \
    tmux \
    vim \
    rsync \
    python3-wstool \
    python3-catkin-tools \
    ros-noetic-cmake-modules \
    protobuf-compiler \
    autoconf

# Install preprocess specific dependencies rospy, rt2_ros, cv_brdige
# RUN apt-get update && apt-get install -y \
#     ros-noetic-rospy \
#     ros-noetic-rt2_ros \
#     ros-noetic-cv_bridge

# Set up the environment for ROS
ENV ROS_WS /home/voxblox_ws
RUN mkdir -p $ROS_WS/src

WORKDIR $ROS_WS

# Source ROS setup file on container launch
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source $ROS_WS/devel/setup.bash" >> ~/.bashrc
ENV DEBIAN_FRONTEND=

# Set the default command to execute
CMD ["bash"]
