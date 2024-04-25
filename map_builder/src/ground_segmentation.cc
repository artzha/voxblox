#include "ground_segmentation.h"

namespace voxblox {

pcl::PointCloud<pcl::PointXYZ>::Ptr extractPlanes(
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
    pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients,
    float distanceThreshold, float angle, int maxIterations) {
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMaxIterations(maxIterations);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(distanceThreshold);

  // We want to find a plane perpendicular to the XY plane
  Eigen::Vector3f axis = Eigen::Vector3f(0.0, 0.0, 1.0);
  seg.setAxis(axis);
  seg.setEpsAngle(
      angle *
      (M_PI / 180.0f));  // plane can be within angle degrees of X-Y plane

  // Create pointcloud to publish inliers
  pcl::ExtractIndices<pcl::PointXYZ> extract;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(
      new pcl::PointCloud<pcl::PointXYZ>());

  // Fit a plane
  seg.setInputCloud(inputCloud);
  seg.segment(*inliers, *coefficients);

  // Check result
  if (inliers->indices.size() == 0) {
    std::cout << "Could not estimate a planar model for the given dataset."
              << std::endl;
    // break;
  }

  // Extract inliers
  extract.setInputCloud(inputCloud);
  extract.setIndices(inliers);
  extract.setNegative(false);
  // Get the points associated with the planar surface
  extract.filter(*cloud_plane);

  return cloud_plane;
}

}  // namespace voxblox
