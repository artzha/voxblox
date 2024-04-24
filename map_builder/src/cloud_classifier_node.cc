#include "cloud_classifier.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <omp.h>
#include <atomic>
#include <iomanip>  // For std::setw and std::setfill
#include <iostream>

int main(int argc, char** argv) {
  ros::init(argc, argv, "cloud_classifier");
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  ros::Rate loop_rate(10);

  voxblox::CloudClassifier classifier(&nh, &nh_private);
  classifier.publishGroundTruth();

  size_t totalPointClouds = classifier.getNumberOfPointClouds();
  std::atomic<size_t> processed(0);
  int progressBarWidth = 50;  // Width of the progress bar in characters

  // Iterate through the point clouds and classify them
  // #pragma omp parallel for
  for (size_t i = 0; i < classifier.getNumberOfPointClouds(); ++i) {
    std::vector<bool> static_dynamic_flags = classifier.queryPointCloud(i);
    classifier.saveFlags(i, static_dynamic_flags);

    size_t currentProcessed =
        ++processed;  // Atomically increment and capture the new value
    float progress = float(currentProcessed) / totalPointClouds;
    int pos = static_cast<int>(progressBarWidth * progress);

    // #pragma omp critical
    // {
    //     std::cout << "\r[" << std::string(pos, '=') <<
    //     std::string(progressBarWidth - pos, ' ')
    //             << "] " << int(progress * 100.0) << " %  (" <<
    //             currentProcessed << "/" << totalPointClouds << ")";
    //     std::cout.flush();  // Flush the stream
    // }
    loop_rate.sleep();
  }
  std::cout << std::endl;  // End the progress bar

  return 0;
}