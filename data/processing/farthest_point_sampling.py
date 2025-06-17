import numpy as np
import open3d as o3d


def farthest_point_sampling(points, n_samples):
    if n_samples >= len(points):
        return points

    pcd = o3d.geometry.PointCloud()
    points_f32 = points.astype(np.float32)
    pcd.points = o3d.utility.Vector3dVector(points_f32)
    down = pcd.farthest_point_down_sample(n_samples)
    return np.asarray(down.points)
