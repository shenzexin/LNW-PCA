"""ISS/FPFH initialization using Open3D feature correspondence RANSAC."""
import numpy as np


def initialize(source, target, voxel=0.001):
    import open3d as o3d
    registration = o3d.pipelines.registration

    def prepare(x):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x))
        cloud = cloud.voxel_down_sample(voxel)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        keypoints = o3d.geometry.keypoint.compute_iss_keypoints(cloud)
        if len(keypoints.points) < 4:
            raise ValueError('Too few ISS keypoints; adjust voxel size or input cloud.')
        tree = o3d.geometry.KDTreeFlann(cloud)
        indices = [tree.search_knn_vector_3d(p, 1)[1][0] for p in keypoints.points]
        full = registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=100))
        descriptor = registration.Feature()
        descriptor.data = np.asarray(full.data)[:, indices]
        return keypoints, descriptor

    p, pf = prepare(source)
    q, qf = prepare(target)
    result = registration.registration_ransac_based_on_feature_matching(
        p, q, pf, qf, True, 0.03,
        registration.TransformationEstimationPointToPoint(False), 3,
        [registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         registration.CorrespondenceCheckerBasedOnDistance(0.03)],
        registration.RANSACConvergenceCriteria(100000, 0.999))
    if len(result.correspondence_set) < 3:
        raise ValueError('Coarse initialization found too few correspondences.')
    return result.transformation
