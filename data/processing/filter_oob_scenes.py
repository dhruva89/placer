import math

import numpy as np
import open3d as o3d
import ray

# -----------------------------------------------------------------------------
# 1) Load AI2-THOR data
# -----------------------------------------------------------------------------
train = ray.data.read_parquet("/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_normalized_aabb/train")
val = ray.data.read_parquet("/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_normalized_aabb/val")
test = ray.data.read_parquet("/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_normalized_aabb/test")

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


# We'll collect objects + point cloud into one list for visualization


# -----------------------------------------------------------------------------
# 4) Create objects in O3D by applying the same transform to each position/orient
# -----------------------------------------------------------------------------
def contains_oob(row):
    sem_classes = np.asarray(row["semantic_class"]).tolist()
    positions = np.asarray(row["positions"]).tolist()  # [x, y, z] in THOR
    orientations = np.asarray(row["rotations"]).tolist()  # Euler angles [x_deg, y_deg, z_deg] in THOR
    input_idx = np.asarray(row["input_idxs"]).tolist()

    norms = np.asarray(row['norms'])
    centroid = np.asarray(row['centroid']).squeeze()
    vertices = np.asarray(row['vertices']) * norms + centroid
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    # 3) Apply this transform *once* to the entire point cloud
    point_cloud.transform(T_thor_to_o3d)

    geometries = []

    furniture_aabb = np.asarray(row["furniture_aabb"], dtype=np.float32) * norms + centroid
    furniture_pcs = []
    for min_bound, max_bound in furniture_aabb:
        # 1) dimensions of the box
        dims = max_bound - min_bound  # shape (3,)

        # 2) create an axis-aligned box of those dimensions
        box = o3d.geometry.TriangleMesh().create_box(
            width=dims[0],
            height=dims[1],
            depth=dims[2]
        )

        # 3) translate its “zero corner” to the AABB’s minimum corner
        box.translate(min_bound)

        # optional: give each box a random color and compute normals
        box.paint_uniform_color(np.random.rand(3))
        box.compute_vertex_normals()

        furniture_pcs.append(box)

    for sclass, pos_thor, euler_deg_thor, input_idx_thor in zip(
            sem_classes, positions, orientations, input_idx):
        # 1) Convert the position vector:  pos_o3d = T_thor_to_o3d * pos_thor
        pos_thor = np.asarray(pos_thor) * norms + centroid
        pos_4 = np.array([pos_thor[0], pos_thor[1], pos_thor[2], 1])
        pos_o3d_4 = T_thor_to_o3d @ pos_4
        pos_o3d = pos_o3d_4[:3]

        # 2) Convert Euler angles from degrees to a THOR rotation matrix
        euler_rad_thor = [math.radians(a) for a in euler_deg_thor]
        R_obj_thor = o3d.geometry.get_rotation_matrix_from_xyz(euler_rad_thor)

        #    Then apply T_thor_to_o3d's rotation part to get orientation in O3D
        R_thor_to_o3d_3x3 = T_thor_to_o3d[:3, :3]  # just the rotation/flip
        R_obj_o3d = R_thor_to_o3d_3x3 @ R_obj_thor

        # 3) Create a box mesh of given size
        if input_idx_thor >= len(furniture_pcs):
            continue
        box_mesh = furniture_pcs[input_idx_thor]

        # Apply the object’s orientation in O3D
        box_mesh.rotate(R_obj_o3d, center=[0, 0, 0])

        # Translate so its center is at pos_o3d
        box_mesh.translate(pos_o3d)

        geometries.append(box_mesh)

    room_bb = point_cloud.get_axis_aligned_bounding_box()
    room_min = room_bb.get_min_bound()
    room_max = room_bb.get_max_bound()

    # test each furniture mesh
    oob_indices = []
    for idx, box in enumerate(geometries):
        bb = box.get_axis_aligned_bounding_box()
        bmin = bb.get_min_bound()
        bmax = bb.get_max_bound()
        # if any corner of the box AABB lies outside the room AABB, mark it
        if np.any(bmin < room_min) or np.any(bmax > room_max):
            oob_indices.append(idx)

    if not oob_indices:
        return True
    else:
        return False


for ds, label in [(train, "train"), (val, "val"), (test, "test")]:
    info = ds.select_columns(
        ["vertices", "positions", "semantic_class", "rotations", "input_idxs", "scene_id", "norms", "centroid",
         "furniture_aabb"])
    info.filter(lambda row: contains_oob(row)).write_parquet(
        f"/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_normalized_aabb_filtered/{label}")
