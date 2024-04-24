cmake_minimum_required(VERSION 2.8.3)

# Set the policy before defining the project
if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()

project(map_builder)


find_package(PCL REQUIRED)
find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(libs glog gflags)

add_definitions(-std=c++17 -Wall -Wextra -fopenmp)

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME}
  src/cloud_classifier.cc
  src/ground_segmentation.cc
)

############
# BINARIES #
############

cs_add_executable(cloud_classifier
  src/cloud_classifier_node.cc
)
target_link_libraries(cloud_classifier ${PROJECT_NAME} ${libs} ${PCL_LIBRARY_DIRS} stdc++fs)

##########
# EXPORT #
##########
cs_install()
cs_export()


# #################
# # EXPORT PYTHON #
# #################

# # Assuming your Python scripts are in the 'scripts/' directory
# catkin_install_python(PROGRAMS
#   preprocess/coda_to_publisher.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )