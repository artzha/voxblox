#include <iostream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

namespace voxblox {
pcl::PointCloud<pcl::PointXYZ>::Ptr extractPlanes(
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
    pcl::PointIndices::Ptr inliers, 
    pcl::ModelCoefficients::Ptr coefficients,
    float distanceThreshold, float angle, int maxIterations
);

}  // namespace voxblox