#!/usr/bin/env python3
"""
generate_pointcloud.py

An example script to:
  1) Load a ProcTHOR scene from a .scene_instance.json
  2) Identify each "Room#" node and extract its geometry as a single mesh
  3) Check which objects belong to each room (centroid-based in this demo)
  4) Optionally add "fixed" objects to the final point cloud
  5) Write out a .ply file for each room's point cloud
"""

from os.path import dirname

import numpy as np
import prior
import trimesh

################################################################################
# Configuration
################################################################################

prior_dataset = prior.load_dataset("procthor-10k")
prior_ds = {'train': prior_dataset.train, 'val': prior_dataset.val, 'test': prior_dataset.test}

allowed_items = {
    "Toaster": 0, "CoffeeMachine": 1, "Microwave": 2, "Chair": 3,
    "DiningTable": 4, "Statue": 5, "Television": 6, "HousePlant": 7,
    "Painting": 8, "Lamp": 9, "Bed": 10, "Blinds": 11, "Mirror": 12,
    "ArmChair": 13, "FloorLamp": 14, "Ottoman": 15, "PaintingSmall": 8,
    "PaintingMedium": 8, "PaintingLarge": 8, "SmallMirror": 12, "Sofa": 16,
    "Vase": 17, "WallMirror": 8, "Shelf": 18, "Dresser": 19, "Desk": 20,
    "SideTable": 21, "Bench": 22, "DeskLamp": 23, "Poster": 24,
    "TVStand": 25, "CoffeeTable": 26, "ShelvingUnit": 27, "DogBed": 28,
    "WashingMachine": 29, "ClothesDryer": 30
}

fixed_objects = {
    "Sink",
    "StoveBurner",
    "Fridge",
    "CounterTop",
    "Toilet",
    "Bathtub",
    "ShowerDoor",
    "LightSwitch",
    "TowelHolder",
    "Window",
    "BathtubBasin",
    "SinkBasin",
    "ShowerGlass",
    "Doorway",
    "WashingMachine",
    "ClothesDryer",
    "Doorframe",
    "Floor",
    "Wall",
}


################################################################################
# Utility functions
################################################################################

def load_glb(glb_path: str, get_scene: bool = False) -> trimesh.Trimesh | trimesh.Scene:
    """
    Load a .glb file with trimesh. If `get_scene=True`, returns a Scene,
    otherwise returns a single Trimesh (merging sub-geometry).
    """
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB not found at {glb_path}")

    # force='scene' => always load as Scene
    # force='mesh'  => merges geometry into a single mesh
    loaded = trimesh.load(
        glb_path,
        force='scene' if get_scene else 'mesh',
        skip_materials=True
    )
    return loaded


def load_stage_mesh(stage_config, stage_config_path) -> trimesh.Scene:
    """
    Given a stage config name (like 'ProcTHOR_00000.stage_config.json'),
    load its data, find the 'render_asset' .glb, load that glb as a Scene,
    and apply the stage config's up/front transform so the entire scene
    is oriented properly.
    """

    # The stage_config has a "render_asset" field => the .glb file to load
    glb_name = stage_config["render_asset"]  # e.g. "ProcTHOR_00000.glb"
    glb_path = os.path.join(dirname(stage_config_path), glb_name)
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"Stage GLB not found: {glb_path}")

    # Load as a Scene
    stage_scene = load_glb(glb_path, get_scene=True)

    # Reorient the entire scene using the stage_config's 'up'/'front' vectors
    stage_scene = apply_stage_config_transform(stage_scene, stage_config)
    return stage_scene


def apply_stage_config_transform(scene: trimesh.Scene, stage_config: dict) -> trimesh.Scene:
    """
    Reorients a Scene according to the stage_config's 'up' and 'front' vectors.
    Assumes the .glb was authored with canonical axes: up=[0,1,0], front=[0,0,-1].
    """
    # target up/front from config
    target_up = np.array(stage_config["up"], dtype=float)
    target_front = np.array(stage_config["front"], dtype=float)
    target_up /= np.linalg.norm(target_up)
    target_front /= np.linalg.norm(target_front)

    # cross => right
    target_right = np.cross(target_up, target_front)
    if np.linalg.norm(target_right) < 1e-8:
        raise ValueError("Up and front vectors are colinear or invalid.")
    target_right /= np.linalg.norm(target_right)

    # Build rotation matrix
    R_target = np.column_stack([target_right, target_up, target_front])

    # 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R_target

    new_scene = scene.copy()
    new_scene.apply_transform(T)
    return new_scene


def sample_point_cloud(mesh: trimesh.Trimesh, num_samples: int) -> np.ndarray:
    """
    Sample 'num_samples' points from a mesh surface.
    Returns an (N, 3) array. If mesh is empty, returns empty (0,3).
    """

    points, _ = trimesh.sample.sample_surface(mesh, num_samples)
    return points


def create_subscene(scene: trimesh.Scene, node_name: str) -> trimesh.Scene:
    """
    Build a new Scene from 'node_name' plus all of its descendants,
    applying their local->world transforms.
    """

    new_scene = trimesh.Scene()

    # Get children/grandchildren, etc. (descendants in the scene graph)
    sub_nodes = scene.graph.transforms.successors(node_name)
    # sub_nodes might be a list or set, depending on version; ensure it's a set:
    sub_nodes = set(sub_nodes)

    for child_node in sub_nodes:
        # The transform from child_node to the 'world' frame
        transform_to_world, geom_key = scene.graph.get(child_node, 'world')
        if geom_key:
            child_geom = scene.geometry[geom_key]
            new_scene.add_geometry(
                child_geom,
                geom_name=geom_key,
                transform=transform_to_world,
            )

    return new_scene


def break_stage_scene_to_rooms(stage_scene: trimesh.Scene) -> list[tuple[str, trimesh.Trimesh]]:
    """
    Finds all nodes whose name starts with "Room#", extracts them (and descendants)
    as sub-scenes, and returns a list of (room_node_name, combined_trimesh).
    """
    room_nodes = [n for n in stage_scene.graph.nodes if n.startswith("Room#") and 'exterior' not in n]
    room_meshes = []

    for room_node in room_nodes:
        # Build a sub-scene for this room
        sub_scene = create_subscene(stage_scene, room_node)

        # Merge into one TriMesh with all transforms applied
        combined_mesh = sub_scene.dump(concatenate=True)
        room_meshes.append((room_node, combined_mesh))

    return room_meshes


def generate_property_data(property_points: dict[str, (list[np.ndarray], list[np.ndarray])],
                           output_dir: str,
                           property_raw_ground_truths):
    """
    Takes a dict: room_index -> [list of point arrays].
    Concatenates them for each room, then writes a .ply file via open3d.
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, (room_arrays, furniture_pcs) in property_points.items():

        gt = property_raw_ground_truths[idx]
        combined_points = np.concatenate(room_arrays, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        output_ply = os.path.join(output_dir, f"{idx}_pointcloud.ply")
        o3d.io.write_point_cloud(output_ply, pcd, write_ascii=True)
        mapping = {}
        for idx2, (asset_id, furniture) in enumerate(furniture_pcs):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(furniture)
            mapping[asset_id] = idx2
            output_ply = os.path.join(output_dir, f"{idx}_furniture|{idx2}_pointcloud.ply")
            o3d.io.write_point_cloud(output_ply, pcd, write_ascii=True)

        objects = []
        for input_idx, semantic_class, position, rotation, size in zip(gt['input_idx'], gt['semantic_classes'],
                                                                       gt['positions'],
                                                                       gt['rotations'], gt['sizes']):
            objects.append(
                {'semantic_class': semantic_class,
                 'input_idx': mapping[input_idx],
                 'position': [float(-position[0]), float(position[1]), float(position[2])],
                 'size': list([float(val) for val in size]),
                 'rotation': list([float(val) for val in rotation])})

        file_name = f"{output_dir}/{idx}.json"

        print(f"Saving JSON file: {file_name}")
        with open(file_name, "w") as json_file:
            json.dump(objects, json_file, indent=4)  # `indent=4` makes the JSON readable


import json
import os
import numpy as np
import trimesh
import open3d as o3d


def load_config(asset_id):
    """ Load the config file for a given asset. """
    config_path = f'/Volumes/Seagate/ai2thor/ai2thor-hab/configs/objects/{asset_id}.object_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def recurse_object(room_id, objects, semantic_classes, input_idx, positions, rotations, sizes, fixed_meshes,
                   seen_allowed_asset_id, allowed_objects_pc):
    """
    Recursively extract object information for a given room, incorporating front and up vectors.
    """
    for obj in objects:
        obj_room_id = obj["id"].split('|')[1] if obj["id"].split('|')[1].isdigit() else obj["id"].split('|')[2]

        if obj_room_id == room_id:
            semantic_class = obj["id"].split("|")[0]
            asset_id = obj["assetId"]
            config = load_config(asset_id)

            up_vector, front_vector = (np.array(config["up"]), np.array(config["front"])) if config else (
                np.array([0, 1, 0]), np.array([0, 0, -1]))

            position = (-obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
            rotation = (obj["rotation"]["x"], -obj["rotation"]["y"], -obj["rotation"]["z"])

            if semantic_class in fixed_objects:
                path = f'/Volumes/Seagate/ai2thor/ai2thor-hab/assets/objects/{asset_id}.glb'
                allowed_obj_scene = trimesh.load(path)
                if 'mesh' in allowed_obj_scene.graph:
                    transform, geom_key = allowed_obj_scene.graph.get('mesh', 'world')
                    if geom_key is not None:
                        obj_mesh = allowed_obj_scene.geometry[geom_key]
                    else:
                        obj_mesh = allowed_obj_scene.dump(concatenate=True)
                else:
                    obj_mesh = allowed_obj_scene.dump(concatenate=True)

                # Compute right vector and rotation matrix
                right_vector = np.cross(up_vector, front_vector)
                rotation_matrix_3x3 = np.column_stack((right_vector, up_vector, front_vector))

                # Apply 180-degree Y-axis rotation
                rotation_matrix_3x3 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]) @ rotation_matrix_3x3

                # Convert to 4x4 transformation matrix and apply it
                rotation_matrix_4x4 = np.eye(4)
                rotation_matrix_4x4[:3, :3] = rotation_matrix_3x3
                obj_mesh.apply_transform(rotation_matrix_4x4)

                # Apply object rotation
                obj_mesh.apply_transform(trimesh.transformations.euler_matrix(*np.radians(rotation), axes='sxyz'))

                # Center object at computed position
                centroid = obj_mesh.bounds.mean(axis=0)
                obj_mesh.apply_translation(-centroid + position)
                fixed_meshes.append(obj_mesh)

            elif semantic_class in allowed_items:
                # Export meshes for specific objects
                path = f'/Volumes/Seagate/ai2thor/ai2thor-hab/assets/objects/{asset_id}.glb'
                allowed_obj_scene = trimesh.load(path)
                if 'mesh' in allowed_obj_scene.graph:
                    transform, geom_key = allowed_obj_scene.graph.get('mesh', 'world')
                    if geom_key is not None:
                        obj_mesh = allowed_obj_scene.geometry[geom_key]
                    else:
                        obj_mesh = allowed_obj_scene.dump(concatenate=True)
                else:
                    obj_mesh = allowed_obj_scene.dump(concatenate=True)

                semantic_class = allowed_items[semantic_class]
                position = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
                rotation = (
                    obj["rotation"]["x"], -obj["rotation"]["y"], -obj["rotation"]["z"])  # Keep original rotation handling

                # Compute right vector and rotation matrix
                right_vector = np.cross(up_vector, front_vector)
                rotation_matrix_3x3 = np.column_stack((right_vector, up_vector, front_vector))

                # Apply 180-degree Y-axis rotation
                rotation_matrix_3x3 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]) @ rotation_matrix_3x3

                # Convert to 4x4 transformation matrix and apply it
                rotation_matrix_4x4 = np.eye(4)
                rotation_matrix_4x4[:3, :3] = rotation_matrix_3x3
                obj_mesh.apply_transform(rotation_matrix_4x4)

                # Center object at computed position
                centroid = obj_mesh.bounds.mean(axis=0)
                obj_mesh.apply_translation(-centroid)

                obj_pc = sample_point_cloud(obj_mesh, num_samples=1024)

                if asset_id not in seen_allowed_asset_id:
                    seen_allowed_asset_id.add(asset_id)
                    allowed_objects_pc.append((asset_id, obj_pc))

                size = obj_mesh.extents
                semantic_classes.append(semantic_class)
                positions.append(position)
                rotations.append(rotation)
                sizes.append(size)
                input_idx.append(asset_id)

            if 'children' in obj:
                recurse_object(room_id, obj['children'], semantic_classes, input_idx, positions, rotations, sizes,
                               fixed_meshes, seen_allowed_asset_id,
                               allowed_objects_pc)


################################################################################
# Main processing function
################################################################################
def process_scene(scene_name: str,
                  proc_thor_dir: str,
                  dataset_root: str,
                  split: str,
                  output_dir: str = "/Volumes/Seagate/ai2thor/ai2thor-hab/processed"):
    """
    Loads a ProcTHOR .scene_instance.json, breaks it into rooms,
    classifies which objects are in each room, optionally includes
    fixed objects in the final point clouds, then writes .plys.
    """

    scene_id = scene_name.split('-')[-1]

    print(f"Processing scene: {proc_thor_dir}/{scene_name}")
    configs_dir = os.path.join(dataset_root, "configs")

    scene_json = os.path.join(configs_dir, "scenes", "ProcTHOR", proc_thor_dir, f"{scene_name}.scene_instance.json")
    if not os.path.exists(scene_json):
        raise FileNotFoundError(f"Scene JSON not found: {scene_json}")

    with open(scene_json, "r") as f:
        scene_data = json.load(f)

    # Identify which stage config => which .glb => load as a Scene
    stage_instance = scene_data["stage_instance"]
    stage_config_name = stage_instance["template_name"] + ".stage_config.json"
    configs_dir = os.path.join(dataset_root, "configs")
    stage_config_path = os.path.join(configs_dir, stage_config_name)
    if not os.path.exists(stage_config_path):
        raise FileNotFoundError(f"Stage config not found: {stage_config_path}")

    with open(stage_config_path, "r") as f:
        stage_config = json.load(f)
    stage_scene = load_stage_mesh(stage_config, stage_config_path)

    # Break into separate (room_node_name, trimesh) pairs
    room_meshes = break_stage_scene_to_rooms(stage_scene)

    # For collecting final point arrays
    property_points = {}

    # (Optional) for debugging which objects ended up in which room
    room_to_object_semantic_ids = {}
    property_raw_ground_truths = {}

    for (room_node, room_mesh) in room_meshes:
        idx = room_node.split("#")[1]
        room_id = idx.split("|")[-1]
        room_to_object_semantic_class_and_meshes = []
        # Start with the mesh for the room itself (30k samples)
        room_pts = [sample_point_cloud(room_mesh, num_samples=30000)]
        room_to_object_semantic_ids[idx] = []

        prior_ds_split = prior_ds[split]
        scene_objects = json.loads(prior_ds_split.select([int(scene_id)]).data[0].decode('utf-8'))['objects']
        semantic_classes, positions, rotations, sizes, fixed_meshes, allowed_items_pc, input_idx, seen_allowed_asset_id = [], [], [], [], [], [], [], set()
        recurse_object(room_id, scene_objects, semantic_classes, input_idx, positions, rotations, sizes, fixed_meshes,
                       seen_allowed_asset_id,
                       allowed_items_pc)

        for fixed_mesh in fixed_meshes:
            fixed_obj_points = sample_point_cloud(fixed_mesh, num_samples=2000)
            room_pts.append(fixed_obj_points)

        property_raw_ground_truths[idx] = {'semantic_classes': semantic_classes, 'positions': positions, 'sizes': sizes,
                                           'rotations': rotations, 'input_idx': input_idx}
        property_points[idx] = (room_pts, allowed_items_pc)

    scene_output_dir = os.path.join(output_dir, proc_thor_dir, scene_name)
    generate_property_data(property_points, scene_output_dir, property_raw_ground_truths)
