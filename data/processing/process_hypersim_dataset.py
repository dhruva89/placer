import asyncio
import gc
import json
import os

import h5py
import httpx
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)

from data.processing.farthest_point_sampling import farthest_point_sampling

# Transformation matrix to convert from ml-Hypersim (Z-up) to AI2-THOR (Y-up)
T = np.array([[1, 0, 0],
              [0, 0, 1],
              [0, -1, 0]])

DATASET_PATH = "/Volumes/Seagate/evermotion_dataset/scenes"


def rotation_matrix_to_quaternion_yup(R_mat):
    """
    Converts a rotation matrix (assumed to be in AI2‑THOR (Y‑up) coordinates)
    to a quaternion. The quaternion is returned in [x, y, z, w] format.
    """
    return R.from_matrix(R_mat).as_quat().tolist()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
)
async def _do_segment(client, payload):
    resp = await client.post('http://localhost:8000/segment', json=payload, timeout=10.0)
    resp.raise_for_status()
    return resp.json()


async def segment_points(property_points):
    async with httpx.AsyncClient() as client:
        payload = {'pc': property_points.tolist()}
        data = await _do_segment(client, payload)

    number_of_rooms = data['count']
    return number_of_rooms


def normalize_furniture_points(points, quaternion, meters_per_asset_unit, unique_obj_mesh_id, output_path):
    # Downsample the points (1024 points max)
    downsampled_obj_points = farthest_point_sampling(points, 1024)

    # Create point cloud object
    obj_pc = o3d.geometry.PointCloud()
    obj_pc.points = o3d.utility.Vector3dVector(downsampled_obj_points)

    # Undo rotation
    qx, qy, qz, qw = quaternion
    R_world_from_model = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
    R_model_from_world = R_world_from_model.T  # inverse
    obj_pc.rotate(R_model_from_world, center=(0, 0, 0))

    # Center
    centroid = obj_pc.get_center()
    obj_pc.translate(-centroid)

    # Scale from asset‐units into meters:
    obj_pc.scale(meters_per_asset_unit, center=(0, 0, 0))

    return obj_pc


async def process_scene(scene_path, output_path):
    """
    Processes the entire scene by combining all trajectories and generating the outputs:
        1. A single .ply file for the space (entire room combined).
        2. Aggregated .ply files for every unique piece of furniture in the scene.
        3. A .json file with metadata for furniture pieces.
    """
    os.makedirs(output_path, exist_ok=False)
    # Prepare paths and data storage
    trajectories_path = os.path.join(scene_path, "images")

    all_furniture_objects = {}  # Dictionary to store unique furniture instances

    # Process 3D bounding boxes from the mesh data
    mesh_path = os.path.join(scene_path, "_detail", "mesh")
    extents_file = os.path.join(mesh_path,
                                "metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5")
    orientations_file = os.path.join(mesh_path,
                                     "metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5")
    positions_file = os.path.join(mesh_path,
                                  "metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5")

    with h5py.File(extents_file, "r") as f:
        obj_extents_data = np.array(f["dataset"])

    with h5py.File(positions_file, "r") as f:
        obj_positions_data = np.array(f["dataset"])

    with h5py.File(orientations_file, "r") as f:
        obj_orientations_data = np.array(f["dataset"])

    metadata_node_strings_file = os.path.join(scene_path, "_detail", "metadata_node_strings.csv")
    metadata_node_strings = pd.read_csv(metadata_node_strings_file)

    mesh_id_mapping = {}

    metadata_asset_units_file = os.path.join(scene_path, "_detail", "metadata_scene.csv")
    metadata_asset_units = pd.read_csv(metadata_asset_units_file)
    meters_per_asset_unit = float(metadata_asset_units.loc[metadata_asset_units[
                                                               'parameter_name'] == 'meters_per_asset_unit', 'parameter_value'].values[
                                      0])

    merged_points = None
    unique_furniture_points = {}

    # Iterate through all trajectories in the scene
    for trajectory_folder in os.listdir(trajectories_path):
        if not trajectory_folder.startswith("scene_cam_") or not trajectory_folder.endswith("_geometry_hdf5"):
            continue

        cam_folder = os.path.join(trajectories_path, trajectory_folder)

        position_files = [file for file in sorted(os.listdir(cam_folder))
                          if file.endswith("position.hdf5") and not file.startswith("._")]
        semantic_files = [file for file in sorted(os.listdir(cam_folder))
                          if file.endswith("semantic.hdf5") and not file.startswith("._")]
        semantic_instance_files = [file for file in sorted(os.listdir(cam_folder))
                                   if file.endswith("semantic_instance.hdf5") and not file.startswith("._")]
        render_entity_id_files = [file for file in sorted(os.listdir(cam_folder))
                                  if file.endswith("render_entity_id.hdf5") and not file.startswith("._")]

        # Iterate through all frames in the current trajectory
        for position_file, semantic_file, semantic_instance_file, render_entity_id_file in zip(position_files,
                                                                                               semantic_files,
                                                                                               semantic_instance_files,
                                                                                               render_entity_id_files):

            position_file = os.path.join(cam_folder, position_file)
            semantic_file = os.path.join(cam_folder, semantic_file)
            semantic_instance_file = os.path.join(cam_folder, semantic_instance_file)
            render_entity_id_file = os.path.join(cam_folder, render_entity_id_file)

            # Load data from HDF5 files
            with h5py.File(position_file, "r") as f:
                position_data = np.array(f["dataset"])

            with h5py.File(semantic_file, "r") as f:
                semantic_data = np.array(f["dataset"])

            with h5py.File(semantic_instance_file, "r") as f:
                semantic_instance_data = np.array(f["dataset"])

            with h5py.File(render_entity_id_file, "r") as f:
                render_entity_id_data = np.array(f["dataset"])

            # Semantic mask and extract room points
            room_semantic_mask = np.isin(semantic_data, list(FIXED_OBJECTS.keys()))
            point_cloud = position_data[room_semantic_mask]

            # Drop invalid points
            valid_mask = np.isfinite(point_cloud).all(axis=1)
            point_cloud = point_cloud[valid_mask]

            # Add to the global scene-wide point cloud
            if merged_points is None:
                merged_points = point_cloud
            else:
                merged_points = np.vstack((merged_points, point_cloud))
            del point_cloud

            # Process allowed furniture objects
            allowed_objects_mask = np.isin(semantic_data, list(ALLOWED_OBJECTS.keys()))
            allowed_obj_instances = set(semantic_instance_data[allowed_objects_mask].reshape(-1).tolist())

            for obj_instance in allowed_obj_instances:
                indices = np.where(semantic_instance_data == obj_instance)
                if indices[0].size == 0:
                    continue

                instance_mask = np.isin(semantic_instance_data, obj_instance) & allowed_objects_mask
                obj_points = position_data[instance_mask]
                render_entity_id = render_entity_id_data[instance_mask][0]
                mesh_id = metadata_node_strings.loc[
                    (metadata_node_strings['node_id'] == render_entity_id) & (metadata_node_strings[
                                                                                  'path'] == '<root>.geometry'), 'string'].values[
                    0]

                if mesh_id not in mesh_id_mapping:
                    mesh_id_mapping[mesh_id] = len(mesh_id_mapping)

                unique_obj_mesh_id = mesh_id_mapping[mesh_id]

                old_rotation = obj_orientations_data[obj_instance]

                furniture_points = normalize_furniture_points(obj_points,
                                                              rotation_matrix_to_quaternion_yup(old_rotation),
                                                              meters_per_asset_unit,
                                                              unique_obj_mesh_id, output_path)

                if unique_obj_mesh_id not in unique_furniture_points:
                    unique_furniture_points[unique_obj_mesh_id] = furniture_points
                else:
                    unique_furniture_points[unique_obj_mesh_id].points.extend(furniture_points.points)

                new_rotation = T @ old_rotation @ T.T
                quaternion = rotation_matrix_to_quaternion_yup(new_rotation)

                # Aggregate points for the same furniture instance
                if obj_instance not in all_furniture_objects:
                    all_furniture_objects[obj_instance] = {
                        'mesh_id': None,
                        'semantic_class': None,
                        'position': None,
                        'size': None,
                        'quaternion': None,
                    }

                    model_semantic_id = ALLOWED_OBJECTS[semantic_data[indices][0]]

                    old_position = obj_positions_data[obj_instance]  # [x, y, z] in ml-Hypersim
                    new_position = (T @ old_position.reshape(3, 1)).flatten()

                    old_extents = obj_extents_data[obj_instance]  # [x, y, z]
                    new_extents = [old_extents[0], old_extents[2], old_extents[1]]

                    all_furniture_objects[obj_instance]['mesh_id'] = unique_obj_mesh_id
                    all_furniture_objects[obj_instance]['semantic_class'] = model_semantic_id
                    all_furniture_objects[obj_instance]['position'] = new_position.tolist()
                    all_furniture_objects[obj_instance]['size'] = new_extents
                    all_furniture_objects[obj_instance]['quaternion'] = quaternion

            del position_data, semantic_data, semantic_instance_data, render_entity_id_data
            gc.collect()

    # Merge all room points into a single scene point cloud (and convert to meters)
    if merged_points is not None:
        # Convert to AI2-THOR coordinate system
        merged_points = (T @ merged_points.T).T

        # scale from asset‐units into meters before sampling
        merged_points *= meters_per_asset_unit
        merged_points = farthest_point_sampling(merged_points, 4096)
        task = asyncio.create_task(segment_points(merged_points))
        number_of_rooms = await task
        if number_of_rooms != 1:
            print(f"Skipping {scene_path}")
            os.rmdir(output_path)
            return

        # Create scene point cloud and save as .ply
        room_pcd = o3d.geometry.PointCloud()
        room_pcd.points = o3d.utility.Vector3dVector(merged_points)
        scene_ply_path = os.path.join(output_path, "scene_space.ply")
        o3d.io.write_point_cloud(scene_ply_path, room_pcd)

    for unique_obj_mesh_id, furniture_pc in unique_furniture_points.items():
        furniture_ply_path = os.path.join(output_path, f"furniture_{unique_obj_mesh_id}.ply")
        if len(np.asarray(furniture_pc.points)) > 1024:
            furniture_pc = furniture_pc.farthest_point_down_sample(1024)
        o3d.io.write_point_cloud(furniture_ply_path, furniture_pc)

    # Save scene metadata as JSON
    metadata_output = [
        {
            'mesh_id': obj_data['mesh_id'],
            'semantic_class': obj_data['semantic_class'],
            'position': (np.array(obj_data['position']) * meters_per_asset_unit).tolist(),
            'size': (np.array(obj_data['size']) * meters_per_asset_unit).tolist(),
            'quaternion': obj_data['quaternion']
        }
        for obj_data in all_furniture_objects.values()
    ]
    scene_metadata_path = os.path.join(output_path, "scene_metadata.json")
    with open(scene_metadata_path, "w") as json_file:
        json.dump(metadata_output, json_file, indent=4)

    print(f"Scene processing completed. Outputs stored in {output_path}.")


# The remaining code that sets up file paths, loads metadata, and calls process_camera_trajectory remains unchanged.

# Read semantic labels and other dataset splits (unchanged)
nyu_40 = pd.read_csv(
    '/Users/dhruvabharadwaj/StageVista/ml-hypersim/code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv')[
    ['semantic_id ', ' semantic_name  ']]
semantic_id_to_class_label = {k: v.strip() for k, v in zip(nyu_40.iloc[:, 0], nyu_40.iloc[:, 1])}

df = pd.read_csv(
    "/Users/dhruvabharadwaj/StageVista/ml-hypersim/evermotion_dataset/analysis/metadata_camera_trajectories.csv")
home_office_animations = df[df['Scene type'].str.contains('Office \\(home\\)')]['Animation'].tolist()
living_room_animations = df[df['Scene type'].str.contains('Living room')]['Animation'].tolist()
kitchen_animations = df[df['Scene type'].str.contains('Kitchen')]['Animation'].tolist()
bathroom_animations = df[df['Scene type'].str.contains('Bathroom')]['Animation'].tolist()
bedroom_animations = df[df['Scene type'].str.contains('Bedroom')]['Animation'].tolist()

shortlisted_animations = set()
shortlisted_animations.update(home_office_animations)
shortlisted_animations.update(living_room_animations)
shortlisted_animations.update(kitchen_animations)
shortlisted_animations.update(bathroom_animations)
shortlisted_animations.update(bedroom_animations)

broken_scenes = ['ai_012_007', 'ai_013_001', 'ai_023_008_cam_01', 'ai_026_020', 'ai_023_009', 'ai_038_007',
                 'ai_023_006', 'ai_026_013', 'ai_026_018', 'ai_003_001', 'ai_004_009', 'ai_023_004', 'ai_044_004']

df['scene'] = df['Animation'].apply(lambda val: "_".join(val.split('_')[:-2]))
valid_scenes = set(df['scene'].tolist())

animations_to_remove = set()
for item in broken_scenes:
    for animation in shortlisted_animations:
        if item in animation:
            animations_to_remove.add(animation)

for animation in shortlisted_animations:
    splits = animation.split('_')
    volume = splits[1]
    scene = splits[2]
    folder = f'ai_{volume}_{scene}'
    if folder not in valid_scenes:
        animations_to_remove.add(animation)

shortlisted_animations -= animations_to_remove

shortlisted_scenes = ["_".join(animation.split('_')[:-2]) for animation in shortlisted_animations]

set_splits = pd.read_csv(
    '/Users/dhruvabharadwaj/StageVista/ml-hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv')

train_splits = set_splits[set_splits['split_partition_name'] == 'train']
val_splits = set_splits[set_splits['split_partition_name'] == 'val']
test_splits = set_splits[set_splits['split_partition_name'] == 'test']

train_animations = set(train_splits['scene_name'].tolist()).intersection(
    shortlisted_scenes)
val_animations = set(val_splits['scene_name']).intersection(
    shortlisted_scenes)
test_animations = set(test_splits['scene_name'].tolist()).intersection(
    shortlisted_scenes)

datasets = [train_animations, val_animations, test_animations]

FIXED_OBJECTS = {
    1: "wall",
    2: "floor",
    8: "door",
    9: "window",
    12: "counter",
    22: "ceiling",
    24: "refrigerator",
    33: "toilet",
    34: "sink",
    36: "bathtub",
    38: "otherstructure"
}

ALLOWED_OBJECTS = {
    3: 18,
    4: 10,
    5: 3,
    6: 16,
    7: 4,
    10: 27,
    11: 8,
    13: 11,
    14: 20,
    15: 18,
    17: 19,
    19: 12,
    25: 6,
    32: 21,
    35: 9
}


def process_scene_sync(args):
    scene_path, output_path = args
    return asyncio.run(process_scene(scene_path, output_path))


if __name__ == '__main__':
    output_dir = '/Volumes/Seagate/evermotion_dataset/processed'

    for dataset in datasets:
        # with ProcessPoolExecutor(max_workers=1) as exe:
        args_list = [
            (os.path.join(DATASET_PATH, folder),
             os.path.join(output_dir, folder))
            for folder in sorted(dataset)
        ]
        for args in args_list:
            # exe.submit(process_scene_sync, args)
            process_scene_sync(args)
