"""
This function loads in a static mesh from a .ply and builds an octomap from it. It saves
this octomap for later and then loads in point clouds from the user specified directory.
For each point in the point cloud, it performs a KNN search to find the nearest points.
If there are at least K points with a distance threshold, it is considered a static point.
Otherwise, it is considered a dynamic point. It then saves the mask indicating the
static and dynamic points.
"""
