#ifndef VOXBLOX_ROS_CLOUD_CLASSIFIER_H_
#define VOXBLOX_ROS_CLOUD_CLASSIFIER_H_

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_search.h>
#include <pcl/io/ply_io.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <regex> 
#include <filesystem> // requires gcc version >= 8

namespace voxblox {

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

struct CODaFilenameComparator {
    bool operator()(const std::string& filename1, const std::string& filename2) const {
        // Regular expression to extract numbers from the format "3d_comp_os1_{seq}_{frame}"
        std::regex re("(?:.*/)?3d_comp_os1_(\\d+)_(\\d+)");
        std::smatch match1, match2;

        // Extracting numbers from the first filename
        if (!std::regex_search(filename1, match1, re)) {
            std::cerr << "Failed to parse " << filename1 << std::endl;
            return false;
        }

        // Extracting numbers from the second filename
        if (!std::regex_search(filename2, match2, re)) {
            std::cerr << "Failed to parse " << filename2 << std::endl;
            return false;
        }

        // Convert extracted strings to integers
        int seq1 = std::stoi(match1[1]);
        int frame1 = std::stoi(match1[2]);
        int seq2 = std::stoi(match2[1]);
        int frame2 = std::stoi(match2[2]);

        // First compare seq, if they are the same then compare frame
        if (seq1 != seq2) return seq1 < seq2;
        return frame1 < frame2;
    }
};

class CloudClassifier {
public:
    CloudClassifier(ros::NodeHandle* nh, ros::NodeHandle* nh_private);

    std::vector<bool> queryPointCloud(size_t index, bool publish);

    void publishGroundTruth();

private:
    ros::NodeHandle* nh_;
    ros::NodeHandle* nh_private_;
    PointCloud::Ptr mesh_;
    pcl::octree::OctreePointCloudSearch<PointT> octree;
    std::string cloudDirectory_;
    std::string meshPath_;
    float resolution = 0.1f; // Default resolution for the octree
    int K = 3;               // Number of nearest neighbors for KNN
    float threshold = 0.1f;  // Distance threshold for determining static points
    ros::Publisher pub_;
    ros::Publisher gt_pub_;
    std::vector<std::string> binPaths_;

    void loadBinPaths(const std::string& binDirectory);
    void publishClassifiedPointCloud(
        const PointCloud::Ptr& cloud, const std::vector<bool>& classification
    );
    void loadPointCloud(const std::string& filepath, PointCloud::Ptr cloud);
};

}  // namespace voxblox


#endif  // VOXBLOX_ROS_TSDF_SERVER_H_