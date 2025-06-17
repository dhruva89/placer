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

import json
import os
from os.path import dirname

import numpy as np
import open3d as o3d
import prior
import trimesh
from transforms3d.quaternions import quat2mat

################################################################################
# Configuration
################################################################################

# Which objects do we consider "fixed" and potentially include in the point cloud?
FIXED_OBJECTS = {
    7: "Sink",
    17: "StoveBurner",
    18: "Fridge",
    27: "CounterTop",
    62: "Toilet",
    64: "Bathtub",
    70: "ShowerDoor",
    135: "Window",
    136: "BathtubBasin",
    137: "SinkBasin",
    133: "ShowerGlass",
    163: "Doorway",
    164: "WashingMachine",
    165: "ClothesDryer",
    166: "Doorframe",
    159: "Floor",
    162: "Wall"
}
# Whether to include the fixed objects' points in each room’s final .ply
INCLUDE_FIXED = True

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


def load_object_mesh(object_config, object_assets_dir: str) -> trimesh.Trimesh:
    """
    Given an object config name (like 'Apple.object_config.json'),
    load the config => find 'render_asset' => load that glb as a single mesh.
    """

    glb_name = object_config["render_asset"]  # e.g. "Apple.glb"
    glb_path = os.path.join(object_assets_dir, glb_name)
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"Object GLB not found: {glb_path}")

    mesh = load_glb(glb_path, get_scene=False)  # single Trimesh
    return mesh


def apply_transform(mesh: trimesh.Trimesh,
                    translation: list[float],
                    rotation_q: list[float],
                    scale: list[float]) -> trimesh.Trimesh:
    """
    Applies scale->rotate->translate to a trimesh mesh.
    rotation_q is [x, y, z, w].
    scale is [sx, sy, sz].
    translation is [tx, ty, tz].
    """

    R = quat2mat(rotation_q)  # 3x3

    # Scale matrix
    S = np.eye(4)
    S[0, 0] = scale[0]
    S[1, 1] = scale[1]
    S[2, 2] = scale[2]

    # Combine S and R => S*R in the top-left 3x3
    SR = np.eye(4)
    SR[:3, :3] = S[:3, :3] @ R

    # Translation
    T = np.eye(4)
    T[:3, 3] = translation

    # Final transform => T * (S*R)
    final_transform = T @ SR

    new_mesh = mesh.copy()
    new_mesh.apply_transform(final_transform)
    return new_mesh


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


def get_object_mesh(obj_instance: dict,
                    object_config,
                    object_assets_dir: str) -> trimesh.Trimesh:
    """
    Loads the base mesh from the object's template config,
    then applies translation/rotation/scale.
    """
    translation = obj_instance["translation"]
    rotation = obj_instance["rotation"]
    scale = obj_instance["non_uniform_scale"]

    base_mesh = load_object_mesh(object_config, object_assets_dir)
    return apply_transform(base_mesh, translation, rotation, scale)


def generate_property_data(property_points: dict[int, list[np.ndarray]],
                           output_dir: str,
                           property_raw_ground_truths):
    """
    Takes a dict: room_index -> [list of point arrays].
    Concatenates them for each room, then writes a .ply file via open3d.
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, room_arrays in property_points.items():

        gt = property_raw_ground_truths[idx]

        combined_points = np.concatenate(room_arrays, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        output_ply = os.path.join(output_dir, f"{idx}_pointcloud.ply")
        print(f"Saving PLY file: {output_ply}")
        o3d.io.write_point_cloud(output_ply, pcd, write_ascii=True)

        objects = []
        for semantic_class, position, rotation, size in zip(gt['semantic_classes'], gt['positions'],
                                                            gt['rotations'], gt['sizes']):
            objects.append(
                {'semantic_class': semantic_class, 'position': [-position[0], position[1], position[2]],
                 'size': list(size),
                 'rotation': list(rotation)})

        file_name = f"{output_dir}/{idx}.json"

        print(f"Saving JSON file: {file_name}")
        with open(file_name, "w") as json_file:
            json.dump(objects, json_file, indent=4)  # `indent=4` makes the JSON readable


def invert_dict(my_dict):
    """
    Inverts a dictionary, swapping keys and values.
    Handles duplicate values by storing corresponding keys in a list.
    """
    return {v: k for k, v in my_dict.items()}


import numpy as np
import trimesh


def is_object_fully_inside_room(object_mesh: trimesh.Trimesh,
                                room_mesh: trimesh.Trimesh) -> bool:
    """
    Returns True if `object_mesh` is fully inside `room_mesh`, given:
      - The object is either entirely inside or entirely outside (no partial)
      - The room mesh may not be watertight
    """

    # 1. BROAD-PHASE: AABB overlap check
    if not _bounding_boxes_overlap(object_mesh, room_mesh):
        # If bounding boxes don't overlap, definitely not inside
        return False

    # 3. POINT-IN-ROOM TEST
    #    Pick a single reference point from the object.
    #    If the object is wholly inside, that point must be inside.
    ref_point = object_mesh.centroid

    # Because the room is not watertight, create a sealed version.
    # We'll use convex_hull here, which is guaranteed closed, but
    # might lose concave detail. Adjust for more accuracy if needed.
    sealed_room = room_mesh.convex_hull

    # Use a ray-cast approach:
    #   If the number of ray intersections is odd => inside
    #   If even => outside
    return _point_in_mesh_by_ray(ref_point, sealed_room)


def _bounding_boxes_overlap(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> bool:
    """
    Quick AABB overlap test for two meshes.
    Returns True if the bounding boxes overlap; False otherwise.
    """
    min_a, max_a = mesh_a.bounds
    min_b, max_b = mesh_b.bounds

    # If one box is completely to the left/right in any dimension => no overlap
    if (max_a[0] < min_b[0]) or (max_b[0] < min_a[0]):
        return False
    if (max_a[1] < min_b[1]) or (max_b[1] < min_a[1]):
        return False
    if (max_a[2] < min_b[2]) or (max_b[2] < min_a[2]):
        return False

    return True


def _point_in_mesh_by_ray(point: np.ndarray, sealed_mesh: trimesh.Trimesh) -> bool:
    """
    Determine if a point is inside a *closed* mesh by ray casting.
    sealed_mesh is assumed manifold & watertight (e.g., convex_hull).
    """
    # Cast one ray from the point in an arbitrary direction (e.g. +X).
    ray_origins = np.array([point])
    ray_directions = np.array([[1.0, 0.0, 0.0]])  # +X direction

    # Intersect the ray with the sealed mesh
    locations, index_ray, index_tri = sealed_mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )

    # Odd => inside, Even => outside
    return (len(locations) % 2 == 1)


################################################################################
# Main processing function
################################################################################

def recurse_object(room_id, objects, semantic_classes, positions, rotations, sizes):
    """
    Recursively extract object information for a given room.
    """
    for obj in objects:
        obj_room_id = obj["id"].split('|')[1]

        if obj_room_id == room_id:
            semantic_class = obj["id"].split("|")[0]
            if semantic_class not in allowed_items.keys():
                continue
            semantic_class = allowed_items[semantic_class]
            position = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
            rotation = (obj["rotation"]["x"], obj["rotation"]["y"], obj["rotation"]["z"])  # Euler angles
            asset_id = obj["assetId"]
            path = f'/Volumes/UnionSine/ai2thor/ai2thor-hab/assets/objects/{asset_id}.glb'
            mesh = o3d.io.read_triangle_mesh(path)
            aabb = mesh.get_axis_aligned_bounding_box()

            # Extract bounding box size
            min_bound = aabb.min_bound
            max_bound = aabb.max_bound
            size = max_bound - min_bound

            semantic_classes.append(semantic_class)
            positions.append(position)
            rotations.append(rotation)
            sizes.append(size)

            if 'children' in obj:
                recurse_object(room_id, obj['children'], semantic_classes, positions, rotations, sizes)


def process_scene(scene_name: str,
                  proc_thor_dir: str,
                  dataset_root: str,
                  split: str,
                  output_dir: str = "/Volumes/UnionSine/ai2thor/ai2thor-hab/processed"):
    """
    Loads a ProcTHOR .scene_instance.json, breaks it into rooms,
    classifies which objects are in each room, optionally includes
    fixed objects in the final point clouds, then writes .plys.
    """

    scene_id = scene_name.split('-')[-1]

    print(f"Processing scene: {proc_thor_dir}/{scene_name}")
    configs_dir = os.path.join(dataset_root, "configs")
    object_assets_dir = os.path.join(dataset_root, "assets", "objects")
    semantic_json_path = os.path.join(configs_dir, "object_semantic_id_mapping.json")

    with open(semantic_json_path, "r") as f:
        semantic_category_to_id = json.load(f)

    semantic_id_to_category = invert_dict(semantic_category_to_id)

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
    semantic_id_to_furniture_class_file = "/Users/dhruvabharadwaj/StageVista/repos/placer/data/processing/semantic_id_to_semantic_class.json"

    with open(semantic_id_to_furniture_class_file, "r") as f:
        semantic_id_to_furniture_class = json.load(f)

    semantic_ids_to_transform_file = "/Users/dhruvabharadwaj/StageVista/repos/placer/data/processing/semantic_ids_to_transform.json"

    with open(semantic_ids_to_transform_file, "r") as f:
        semantic_ids_to_transform = json.load(f)

    for (room_node, room_mesh) in room_meshes:
        idx = room_node.split("#")[1]
        room_id = idx.split("|")[-1]
        room_to_object_semantic_class_and_meshes = []
        # Start with the mesh for the room itself (30k samples)
        room_pts = [sample_point_cloud(room_mesh, num_samples=30000)]
        room_to_object_semantic_ids[idx] = []

        # # For each object in the scene, load + see if it is in this room
        # for obj_instance in scene_data["object_instances"]:
        #     tmpl_name = obj_instance["template_name"] + ".object_config.json"
        #
        #     object_config_path = os.path.join(configs_dir, tmpl_name)
        #     if not os.path.exists(object_config_path):
        #         raise FileNotFoundError(f"Object config not found: {object_config_path}")
        #
        #     with open(object_config_path, "r") as f:
        #         object_config = json.load(f)
        #
        #     obj_mesh = get_object_mesh(obj_instance, object_config, object_assets_dir)
        #     obj_semantic_id = object_config['semantic_id']
        #     obj_semantic_category = semantic_id_to_category[obj_semantic_id]
        #     if is_object_fully_inside_room(obj_mesh, room_mesh):
        #         # Merge similar items
        #         if str(obj_semantic_id) in semantic_ids_to_transform:
        #             obj_semantic_id = semantic_ids_to_transform[str(obj_semantic_id)]
        #
        #         # Track it
        #         room_to_object_semantic_ids[idx].append(obj_semantic_id)
        #
        #         if obj_semantic_category not in FIXED_OBJECTS and str(
        #                 obj_semantic_id) in semantic_id_to_furniture_class.keys():
        #             room_to_object_semantic_class_and_meshes.append(
        #                 (semantic_id_to_furniture_class[str(obj_semantic_id)], obj_mesh))
        #
        #         # If we are including "fixed" objects, check if base_name is in FIXED_OBJECTS
        #         if INCLUDE_FIXED and obj_semantic_category in FIXED_OBJECTS:
        #             # Add 5k points for this object
        #             obj_points = sample_point_cloud(obj_mesh, num_samples=5000)
        #             room_pts.append(obj_points)

        prior_ds_split = prior_ds[split]
        scene_objects = json.loads(prior_ds_split.select([int(scene_id)]).data[0].decode('utf-8'))['objects']
        semantic_classes, positions, rotations, sizes, fixed_meshes = [], [], [], [], []
        recurse_object(room_id, scene_objects, semantic_classes, positions, rotations, sizes)

        property_raw_ground_truths[idx] = {'semantic_classes': semantic_classes, 'positions': positions, 'sizes': sizes,
                                           'rotations': rotations}
        property_points[idx] = room_pts

    scene_output_dir = os.path.join(output_dir, proc_thor_dir, scene_name)
    generate_property_data(property_points, scene_output_dir, property_raw_ground_truths)
