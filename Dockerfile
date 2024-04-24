# Use ROS Noetic official image
FROM ros:noetic-robot

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
    python-is-python3 \
    python3-wstool \
    python3-catkin-tools \
    ros-noetic-cmake-modules \
    ros-noetic-rviz \
    ros-noetic-cv-bridge \
    ros-noetic-pcl-conversions \
    ros-noetic-pcl-ros \
    python3-pip \
    libgoogle-glog-dev \
    libgflags-dev \
    protobuf-compiler \
    libprotobuf-dev \
    autoconf

RUN pip3 install -U tqdm scipy matplotlib scikit-learn numpy-quaternion

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
