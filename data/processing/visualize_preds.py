import json
import math
import random

import numpy as np
import open3d as o3d
import ray

# -----------------------------------------------------------------------------
# 1) Load AI2-THOR data
# -----------------------------------------------------------------------------
ds = ray.data.read_parquet(
    "/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_normalized_sorted/val"
)
print(ds.schema())
info = ds.select_columns(["positions", "semantic_class", "rotations", "input_idxs", "scene_id", "norms", "centroid",
                          "furniture_vertices"]).take(1)[0]
print(info['scene_id'])
norms = info['norms']
centroid = np.asarray(info['centroid']).squeeze()
vertices = np.asarray(ds.select_columns(["vertices"]).take(1)[0]['vertices'], dtype=np.float32) * norms + centroid
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(vertices)

# -----------------------------------------------------------------------------
# 2) Define a SINGLE global transform: Y-up --> Z-up, then reflect about X=0
# -----------------------------------------------------------------------------
# (a) Rotate -90° about X so that Y_thor becomes Z_o3d.
R_90x = o3d.geometry.get_rotation_matrix_from_xyz([math.pi / 2, 0, 0])  # or [-π/2, 0, 0] if needed
T_90x = np.eye(4)
T_90x[:3, :3] = R_90x

# (b) Reflect about X=0, i.e. x -> -x (flip the X coordinate)
# flip_x = np.eye(4)
# flip_x[0, 0] = -1.0  # multiply X by -1

# Combine them.  By default, the transform on the right acts first:
T_thor_to_o3d = T_90x
# T_thor_to_o3d_2  =
# If you see an incorrect orientation, try reversing the multiply order:
# T_thor_to_o3d = T_90x @ flip_x

# 3) Apply this transform *once* to the entire point cloud
point_cloud.transform(T_thor_to_o3d)

# We'll collect objects + point cloud into one list for visualization
geometries = [point_cloud]
color_map = {}  # e.g. semantic_class -> [R, G, B]

row = json.load(
    open('/Users/dhruvabharadwaj/Library/Application Support/JetBrains/PyCharm2024.3/scratches/scratch_66.json',
         mode='r'))

furniture_vertices = np.asarray(ds.select_columns(["furniture_vertices"]).take(1)[0]['furniture_vertices'],
                                dtype=np.float32) * norms + centroid
furniture_pcs = [o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float32)) for vertices in furniture_vertices]


# -----------------------------------------------------------------------------
# 4) Create objects in O3D by applying the same transform to each position/orient
# -----------------------------------------------------------------------------
def augment_ply(row):
    sem_classes = row["semantic_class"][0]
    positions = row["position_pred"][0]
    orientations = row["orientation_pred"][0]  # Euler angles [x_deg, y_deg, z_deg] in THOR
    input_idx = row["input_idx_pred"][0]

    for sclass, pos_thor, euler_rad_thor, input_idx_thor in zip(
            sem_classes, positions, orientations, input_idx):
        # 1) Convert the position vector:  pos_o3d = T_thor_to_o3d * pos_thor
        pos_thor = np.asarray(pos_thor) * norms + centroid
        pos_4 = np.array([pos_thor[0], pos_thor[1], pos_thor[2], 1])
        pos_o3d_4 = T_thor_to_o3d @ pos_4
        pos_o3d = pos_o3d_4[:3]

        # 2) Convert Euler angles from degrees to a THOR rotation matrix
        R_obj_thor = o3d.geometry.get_rotation_matrix_from_xyz(euler_rad_thor)

        #    Then apply T_thor_to_o3d's rotation part to get orientation in O3D
        R_thor_to_o3d_3x3 = T_thor_to_o3d[:3, :3]  # just the rotation/flip
        R_obj_o3d = R_thor_to_o3d_3x3 @ R_obj_thor

        # 3) Create a box mesh of given size
        if input_idx_thor >= len(furniture_pcs):
            continue
        furniture_pc = furniture_pcs[input_idx_thor]
        furniture_points = o3d.utility.Vector3dVector(furniture_pc)
        box_mesh = o3d.geometry.PointCloud()
        box_mesh.points = furniture_points

        # Move its center to the origin
        # extents = box_mesh.get_axis_aligned_bounding_box().get_extent()
        # box_mesh.translate([-0.5 * extents[0], -0.5 * extents[1], -0.5 * extents[2]])

        # Apply the object’s orientation in O3D
        box_mesh.rotate(R_obj_o3d, center=[0, 0, 0])

        # Translate so its center is at pos_o3d
        box_mesh.translate(pos_o3d)

        # Color it
        color = color_map.get(sclass, [random.random(), random.random(), random.random()])
        box_mesh.paint_uniform_color(color)

        geometries.append(box_mesh)


# 5) Build the bounding boxes
augment_ply(row)

# 6) Visualize everything in Open3D
o3d.visualization.draw_geometries(geometries)
