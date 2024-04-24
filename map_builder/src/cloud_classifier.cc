#include "cloud_classifier.h"

#include "ground_segmentation.h"

#include <pcl/common/transforms.h>

namespace fs = std::filesystem;

namespace voxblox {

CloudClassifier::CloudClassifier(ros::NodeHandle* nh, ros::NodeHandle* nh_private) : 
        nh_(nh), nh_private_(nh_private), octree(0.1f) {
    // Load node params
    nh_private->param("cluster", cluster_, 0);
    nh_private->param("infoDir", infoDirectory_, std::string(""));
    nh_private->param("cloudDir", cloudDirectory_, std::string(""));
    nh_private->param("resultDir", resultDir_, std::string(""));
    nh_private->param("meshPath", meshPath_, std::string(""));
    nh_private->param("voxelSize", resolution, 0.1f);
    nh_private->param("kNeighbors", K, 3);
    nh_private->param("kThreshold", threshold, 0.1f);
    nh_private->param("groundThreshold", groundThreshold_, 0.1f);
    nh_private->param("groundAngle", groundAngle_, 12.0f);
    nh_private->param("groundMaxIterations", groundMaxIterations_, 1000);
    nh_private->param("publish", publish_, true);

    mesh_ = PointCloudT::Ptr(new PointCloudT);
    if (pcl::io::loadPLYFile<PointT>(meshPath_, *mesh_) == -1) {
        PCL_ERROR("Couldn't read mesh file.\n");
        throw std::runtime_error("Failed to load mesh file.");
    }

    // Initialize publishers
    pub_ = nh_->advertise<sensor_msgs::PointCloud2>("classified_cloud", 1);
    gt_pub_ = nh_->advertise<sensor_msgs::PointCloud2>("ground_truth", 1);

    // Load bin paths
    loadFrameInfos(infoDirectory_);

    // Publish ground truth point cloud from mesh
    octree.setInputCloud(mesh_);
    octree.addPointsFromInputCloud();
    this->publishGroundTruth();
    
    std::cout << "Initialization Done." << std::endl;
}

void CloudClassifier::publishGroundTruth() {
    std::cout << "Publishing ground truth point cloud from mesh..." << std::endl;
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*mesh_, output);
    output.header.frame_id = "world";
    output.header.stamp = ros::Time::now();
    gt_pub_.publish(output);
    std::cout << "Ground truth point cloud published." << std::endl;
}

void CloudClassifier::loadFrameInfos(const std::string& infoDirectory) {
    std::ifstream clusterFile(infoDirectory + "cluster_labels.txt");
    std::ifstream seqFrameFile(infoDirectory + "global_infos.txt");
    std::ifstream poseFile(infoDirectory + "global_poses.txt");
    
    std::string clusterLine, seqFrameLine, poseLine;

    while (std::getline(clusterFile, clusterLine) && 
           std::getline(seqFrameFile, seqFrameLine) &&
           std::getline(poseFile, poseLine)) {
        std::istringstream seqFrameStream(seqFrameLine);
        std::istringstream poseStream(poseLine);

        int clusterId;
        int seq, frame;
        double x, y, z, r1, r2, r3, r4, r5, r6, r7, r8, r9;

        // Read the cluster ID
        std::istringstream(clusterLine) >> clusterId;

        // Read sequence and frame
        seqFrameStream >> seq >> frame;

        // Read pose data
        poseStream >> r1 >> r2 >> r3 >> r4 >> r5 >> r6 >> r7 >> r8 >> r9 >> x >> y >> z;
        
        // Only proceed if the cluster ID matches your criteria
        if (clusterId == cluster_) {
            // Construct the file path for the point cloud
            std::ostringstream filePath;
            filePath << cloudDirectory_ << seq << "/3d_comp_os1_" << seq << "_" << frame << ".bin";

            // Convert to SE3 pose
            Eigen::Matrix3d rotationMatrix;
            rotationMatrix << r1, r2, r3,
                              r4, r5, r6,
                              r7, r8, r9;
            Eigen::Vector3d translation(x, y, z);
            Eigen::Affine3d pose = Eigen::Affine3d::Identity();
            pose.linear() = rotationMatrix;
            pose.translation() = translation;

            // Add to frame infos
            FrameInfo frameInfo;
            frameInfo.seq = std::to_string(seq);
            frameInfo.frame = std::to_string(frame);
            frameInfo.pc_path = filePath.str();
            frameInfo.pose = pose;
            frameInfos_.push_back(frameInfo);
        }
    }

    // Close files
    clusterFile.close();
    seqFrameFile.close();
    poseFile.close();
    std::cout << "Loaded " << frameInfos_.size() << " frames." << std::endl;
}

std::vector<bool> CloudClassifier::queryPointCloud(size_t index) {
    //1 - Load point cloud and SE3 map transformation
    PointCloudT::Ptr cloud(new PointCloudT);
    loadPointCloud(frameInfos_[index].pc_path, cloud);
    
    auto pose = frameInfos_[index].pose;
    std::vector<bool> static_dynamic_flags(cloud->size(), false);

    //2 - Perform ground segmentation 
    pcl::PointIndices::Ptr shiftedInliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr shiftedCoefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud = extractPlanes(
        cloud, shiftedInliers, shiftedCoefficients, 
        groundThreshold_, groundAngle_, groundMaxIterations_
    );

    //3 Set static points based on shiftedInliers
    for (size_t i = 0; i < cloud->size(); i++) {
        if (std::find(shiftedInliers->indices.begin(), shiftedInliers->indices.end(), i) != shiftedInliers->indices.end()) {
            static_dynamic_flags[i] = true;
        }
    }

    //4 Transform point cloud to map frame
    pcl::transformPointCloud(*cloud, *cloud, pose.cast<float>());

    //4 Octree based KNN filtering
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;

    for (size_t i = 0; i < cloud->size(); ++i) {
        octree.nearestKSearch(cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance);

        int count = 0;
        for (float distance : pointNKNSquaredDistance) {
            if (distance <= threshold * threshold) {
                count++;
            }
        }
        static_dynamic_flags[i] = (count >= K); // True if static, false if dynamic
    }

    PointCloudRGB::Ptr cloud_rgb(new PointCloudRGB);
    pcl::copyPointCloud(*cloud, *cloud_rgb);
    if (publish_) {
        publishClassifiedPointCloud(cloud_rgb, static_dynamic_flags);
    }

    return static_dynamic_flags;
}

void CloudClassifier::saveFlags(size_t index, const std::vector<bool>& flags) {
    // Construct the file path for the flag
    std::string seq = frameInfos_[index].seq;
    std::string frame = frameInfos_[index].frame;
    std::string seqDir = resultDir_ + seq;

    try {
        // This will create all non-existent parent directories along the path
        if (fs::create_directories(seqDir)) {
            std::cout << "Successfully created directories: " << seqDir << std::endl;
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating directories: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    //Construct the file path for the flag
    std::ostringstream filePath;
    filePath << seqDir << "/" << frame << ".bin";

    // Save the flags
    std::ofstream file(filePath.str(), std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filePath.str() << std::endl;
        return;
    }
    for (bool flag : flags) {
        file.write(reinterpret_cast<const char*>(&flag), sizeof(bool));
    }
}

void CloudClassifier::publishClassifiedPointCloud(
    const PointCloudRGB::Ptr& cloud, const std::vector<bool>& classification
) {
    for (size_t i = 0; i < cloud->size(); i++) {
        if (classification[i]) {
            cloud->points[i].r = 0; cloud->points[i].g = 0; cloud->points[i].b = 0; // Black for static
        } else {
            cloud->points[i].r = 255; cloud->points[i].g = 0; cloud->points[i].b = 0; // Red for dynamic
        }
    }

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header.frame_id = "world";
    output.header.stamp = ros::Time::now();
    pub_.publish(output);
}

void CloudClassifier::loadPointCloud(const std::string& filepath, PointCloudT::Ptr cloud) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filepath << std::endl;
        return;
    }

    float intensity = 0;
    while (!file.eof()) {
        PointT point;
        file.read(reinterpret_cast<char*>(&point.x), sizeof(float));
        file.read(reinterpret_cast<char*>(&point.y), sizeof(float));
        file.read(reinterpret_cast<char*>(&point.z), sizeof(float));
        // Drop intensity
        file.read(reinterpret_cast<char*>(&intensity), sizeof(float));

        if(file.gcount() == sizeof(float))
            cloud->push_back(point);
    }

    file.close();
}

}  // namespace voxblox