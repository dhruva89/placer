import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import prior
import ray
import trimesh  # make sure trimesh is installed

# =============================================================================
# Global semantic mapping and fixed object IDs
# =============================================================================
global_semantic_mapping = {
    "Undefined": 0,
    "Apple": 1,
    "AppleSliced": 2,
    "Tomato": 3,
    "TomatoSliced": 4,
    "Bread": 5,
    "BreadSliced": 6,
    "Sink": 7,
    "Pot": 8,
    "Pan": 9,
    "Knife": 10,
    "Fork": 11,
    "Spoon": 12,
    "Bowl": 13,
    "Toaster": 14,
    "CoffeeMachine": 15,
    "Microwave": 16,
    "StoveBurner": 17,
    "Fridge": 18,
    "Cabinet": 19,
    "Egg": 20,
    "Chair": 21,
    "Lettuce": 22,
    "Potato": 23,
    "Mug": 24,
    "Plate": 25,
    "DiningTable": 26,
    "CounterTop": 27,
    "GarbageCan": 28,
    "Omelette": 29,
    "EggShell": 30,
    "EggCracked": 31,
    "StoveKnob": 32,
    "Container": 33,
    "Cup": 34,
    "ButterKnife": 35,
    "PotatoSliced": 36,
    "MugFilled": 37,
    "BowlFilled": 38,
    "Statue": 39,
    "LettuceSliced": 40,
    "ContainerFull": 41,
    "BowlDirty": 42,
    "Sandwich": 43,
    "Television": 44,
    "HousePlant": 45,
    "TissueBox": 46,
    "VacuumCleaner": 47,
    "Painting": 48,
    "WateringCan": 49,
    "Laptop": 50,
    "RemoteControl": 51,
    "Box": 52,
    "Newspaper": 53,
    "TissueBoxEmpty": 54,
    "PaintingHanger": 55,
    "KeyChain": 56,
    "Dirt": 57,
    "CellPhone": 58,
    "CreditCard": 59,
    "Cloth": 60,
    "Candle": 61,
    "Toilet": 62,
    "Plunger": 63,
    "Bathtub": 64,
    "ToiletPaper": 65,
    "ToiletPaperHanger": 66,
    "SoapBottle": 67,
    "SoapBottleFilled": 68,
    "SoapBar": 69,
    "ShowerDoor": 70,
    "SprayBottle": 71,
    "ScrubBrush": 72,
    "ToiletPaperRoll": 73,
    "Lamp": 74,
    "LightSwitch": 75,
    "Bed": 76,
    "Book": 77,
    "AlarmClock": 78,
    "SportsEquipment": 79,
    "Pen": 80,
    "Pencil": 81,
    "Blinds": 82,
    "Mirror": 83,
    "TowelHolder": 84,
    "Towel": 85,
    "Watch": 86,
    "MiscTableObject": 87,
    "ArmChair": 88,
    "BaseballBat": 89,
    "BasketBall": 90,
    "Faucet": 91,
    "Boots": 92,
    "Bottle": 93,
    "DishSponge": 94,
    "Drawer": 95,
    "FloorLamp": 96,
    "Kettle": 97,
    "LaundryHamper": 98,
    "LaundryHamperLid": 99,
    "Lighter": 100,
    "Ottoman": 101,
    "PaintingSmall": 102,
    "PaintingMedium": 103,
    "PaintingLarge": 104,
    "PaintingHangerSmall": 105,
    "PaintingHangerMedium": 106,
    "PaintingHangerLarge": 107,
    "PanLid": 108,
    "PaperTowelRoll": 109,
    "PepperShaker": 110,
    "PotLid": 111,
    "SaltShaker": 112,
    "Safe": 113,
    "SmallMirror": 114,
    "Sofa": 115,
    "SoapContainer": 116,
    "Spatula": 117,
    "TeddyBear": 118,
    "TennisRacket": 119,
    "Tissue": 120,
    "Vase": 121,
    "WallMirror": 122,
    "MassObjectSpawner": 123,
    "MassScale": 124,
    "Footstool": 125,
    "Shelf": 126,
    "Dresser": 127,
    "Desk": 128,
    "SideTable": 129,
    "Pillow": 130,
    "Bench": 131,
    "Cart": 132,
    "ShowerGlass": 133,
    "DeskLamp": 134,
    "Window": 135,
    "BathtubBasin": 136,
    "SinkBasin": 137,
    "CD": 138,
    "Curtains": 139,
    "Poster": 140,
    "HandTowel": 141,
    "HandTowelHolder": 142,
    "Ladle": 143,
    "WineBottle": 144,
    "ShowerCurtain": 145,
    "ShowerHead": 146,
    "TVStand": 147,
    "CoffeeTable": 148,
    "ShelvingUnit": 149,
    "AluminumFoil": 150,
    "DogBed": 151,
    "Dumbbell": 152,
    "TableTopDecor": 153,
    "RoomDecor": 154,
    "Stool": 155,
    "GarbageBag": 156,
    "Desktop": 157,
    "TargetCircle": 158,
    "Floor": 159,
    "ScreenFrame": 160,
    "ScreenSheet": 161,
    "Wall": 162,
    "Doorway": 163,
    "WashingMachine": 164,
    "ClothesDryer": 165,
    "Doorframe": 166
}
global_semantic_counter = 167

disallowed_room_objects = {
    "Bathroom": [
        0,
        4,
        5,
        7,
        10,
        12,
        20,
        24,
        27,
        30
    ],
    "Bedroom": [
        0,
        5
    ],
    "Kitchen": [
        7,
        10,
        12,
        20,
        23,
        24,
        27,
        30
    ],
    "LivingRoom": [
        10,
        23
    ]
}
semantic_id_to_class = {
    "16": 0,
    "17": 1,
    "18": 2,
    "19": 3,
    "21": 4,
    "26": 5,
    "27": 6,
    "44": 7,
    "48": 8,
    "74": 9,
    "76": 10,
    "83": 11,
    "88": 12,
    "95": 13,
    "96": 14,
    "101": 15,
    "102": 16,
    "103": 17,
    "104": 18,
    "114": 19,
    "115": 20,
    "122": 21,
    "126": 22,
    "127": 23,
    "128": 24,
    "129": 25,
    "131": 26,
    "134": 27,
    "135": 28,
    "140": 29,
    "147": 30,
    "148": 31,
    "149": 32,
    "153": 33,
    "154": 34,
    "164": 35
}


# =============================================================================
# Compute a more accurate AABB given an object's position, size and orientation.
# =============================================================================

def compute_aabb(position, size, orientation):
    """
    Compute the axis-aligned bounding box (AABB) for an object.

    - position: (x, y, z)
    - size: (length, breadth, height)
    - orientation: (rx, ry, rz) in degrees (Euler angles)

    The function computes the 8 corners of the oriented bounding box (OBB),
    rotates them by the given Euler angles (order: x, then y, then z),
    translates them by the object's position, and returns the min and max.
    """
    l, b, h = size
    half = np.array([l / 2, b / 2, h / 2])
    # Create the 8 corners relative to the center.
    corners = []
    for dx in [-half[0], half[0]]:
        for dy in [-half[1], half[1]]:
            for dz in [-half[2], half[2]]:
                corners.append(np.array([dx, dy, dz]))

    # Build rotation matrices (angles converted to radians)
    rx, ry, rz = np.radians(orientation)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0],
                   [0, 0, 1]])
    # Combined rotation (adjust order if needed)
    R = Rx @ Ry @ Rz

    transformed = []
    for corner in corners:
        tcorner = R @ corner + np.array(position)
        transformed.append(tcorner)
    transformed = np.array(transformed)
    min_corner = transformed.min(axis=0)
    max_corner = transformed.max(axis=0)
    return [min_corner, max_corner]


# =============================================================================
# Helper to load object config file (if available)
# =============================================================================

def get_object_config(asset_id):
    """
    Given an asset id, check for a config file at the designated directory.
    Returns a dictionary with config info if the file exists; otherwise None.
    """
    config_path = os.path.join("/Volumes/UnionSine/ai2thor/ai2thor-hab/configs/objects",
                               f"{asset_id}.object_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return None


# =============================================================================
# A helper for 2D point-in-polygon check (using shapely)
# =============================================================================

def point_in_polygon_2d(pt, polygon):
    """
    Check whether a 2D point (x, z) lies inside the projection (x,z) of a 3D polygon.
    """
    from shapely.geometry import Point, Polygon
    poly_2d = Polygon([(p["x"], p["z"]) for p in polygon])
    return poly_2d.contains(Point(pt[0], pt[1]))


# =============================================================================
# Sample points from a room's surfaces (including fixed object surfaces)
# ============================================================================


# =============================================================================
# Process a scene (row) from the dataset
# =============================================================================

def process_sample(row):
    """
    For a given scene (row), produce one output dict per room.
    Each output dict contains:
      - room_id
      - room_type
      - scene_id
      - vertices: 1024 points sampled from the surfaces (room walls, floor, ceiling and fixed objects)
      - semantic_classes: list of semantic ids (one per object in the room)
      - position: list of object positions (x,y,z)
      - orientation: list of object rotations (x,y,z)
      - size: list of object sizes (length, breadth, height)
      - aabb: list of accurate AABBs computed from position, size, and orientation
    """
    global global_semantic_mapping, global_semantic_counter  # for updating our mapping
    row_id = row.get("row_id")
    scene_id = row.get("metadata", {}).get("roomSpecId", "unknown_scene")
    out_rows = []
    row_id = f"ProcTHOR-{row.get('split')}-{row_id}"
    for room in row.get("rooms", []):
        room_id = room.get("id")
        if "exterior" in room_id:
            continue

        filter_id = f"{row_id}_{room_id}"
        if filter_id not in old_ids[row['split']]:
            print(f"Filtering out {filter_id}")
            continue

        room_type = room.get("roomType")

        # Filter objects that lie in the room (using the room's floor polygon)
        room_floor = room.get("floorPolygon", [])
        objects_in_room = []
        if len(room_floor) >= 3:
            for obj in row.get("objects", []):
                pos = obj.get("position", {})
                if "x" in pos and "z" in pos:
                    if point_in_polygon_2d((pos["x"], pos["z"]), room_floor):
                        objects_in_room.append(obj)

        semantic_classes = []
        positions = []
        orientations = []
        sizes = []
        aabbs = []
        for obj in objects_in_room:
            asset_id = obj.get("assetId")
            config = get_object_config(asset_id)

            if config is None:
                continue

            semantic_id = config["semantic_id"]

            if str(semantic_id) not in semantic_id_to_class:
                continue

            semantic_class = semantic_id_to_class[str(semantic_id)]
            semantic_classes.append(semantic_class)

            if semantic_class in disallowed_room_objects[room_type]:
                continue

            # Collect position & orientation.
            pos = obj.get("position", {})
            pos_tuple = [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]
            positions.append(pos_tuple)
            rot = obj.get("rotation", {})
            rot_tuple = [rot.get("x", 0), rot.get("y", 0), rot.get("z", 0)]
            orientations.append(rot_tuple)

            config_dir = "/Volumes/UnionSine/ai2thor/ai2thor-hab/configs/objects"
            glb_rel_path = config["render_asset"]
            glb_path = os.path.abspath(os.path.join(config_dir, glb_rel_path))
            mesh = trimesh.load(glb_path, force='mesh')
            size_tuple = mesh.bounding_box.extents
            sizes.append(size_tuple)
            aabb = compute_aabb(pos_tuple, size_tuple, rot_tuple)
            aabbs.append(aabb)

        new_row = {
            "room_id": room_id,
            "row_id": row_id,
            "room_type": room_type,
            "scene_id": scene_id,
            "semantic_classes": np.asarray(semantic_classes, dtype=np.int64).tolist(),  # one semantic id per object
            "position": np.asarray(positions, dtype=np.float32).tolist(),  # object positions (x,y,z)
            "orientation": np.asarray(orientations, dtype=np.float32).tolist(),  # object rotations (x,y,z)
            "size": np.asarray(sizes, dtype=np.float32).tolist(),  # object sizes (length, breadth, height)
            "aabb": np.asarray(aabbs, dtype=np.float32).tolist(),  # computed axis-aligned bounding boxes
        }
        out_rows.append(new_row)
    return out_rows


# =============================================================================
# Process the dataset
# =============================================================================

def read_dataset(dataset_split, split_type):
    json_list = [sample.decode() for sample in dataset_split]
    json_list = [{**json.loads(row_json), 'row_id': idx, 'split': split_type} for idx, row_json in enumerate(json_list)]
    json_data = json.dumps(json_list)
    df = pd.read_json(io.StringIO(json_data))
    return ray.data.from_pandas(df)


prior_dataset = prior.load_dataset("procthor-10k")

train_dataset = read_dataset(prior_dataset.train.data, "Train")
val_dataset = read_dataset(prior_dataset.val.data, "Val")
test_dataset = read_dataset(prior_dataset.test.data, "Test")

old_train_dataset = ray.data.read_parquet('/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_final_processed/train/')
old_val_dataset = ray.data.read_parquet('/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_final_processed/val/')
old_test_dataset = ray.data.read_parquet('/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_final_processed/test/')

old_train_ids = old_train_dataset.select_columns(['scene_id', 'room_id']).map(
    lambda row: {'filter': row['scene_id'] + '_' + row['room_id']}).to_pandas()['filter'].tolist()
old_val_ids = old_val_dataset.select_columns(['scene_id', 'room_id']).map(
    lambda row: {'filter': row['scene_id'] + '_' + row['room_id']}).to_pandas()['filter'].tolist()
old_test_ids = old_test_dataset.select_columns(['scene_id', 'room_id']).map(
    lambda row: {'filter': row['scene_id'] + '_' + row['room_id']}).to_pandas()['filter'].tolist()

old_ids = {
    'Train': set(old_train_ids),
    'Val': set(old_val_ids),
    'Test': set(old_test_ids),

}

processed_train_ds = train_dataset.flat_map(process_sample).sort(['row_id', 'room_id'])
processed_val_ds = val_dataset.flat_map(process_sample).sort(['row_id', 'room_id'])
processed_test_ds = test_dataset.flat_map(process_sample).sort(['row_id', 'room_id'])

processed_train_ids = set(processed_train_ds.select_columns(['row_id', 'room_id']).map(
    lambda row: {'filter': f"{row['row_id']}_{row['room_id']}"}).to_pandas()['filter'].tolist())
processed_val_ids = set(processed_val_ds.select_columns(['row_id', 'room_id']).map(
    lambda row: {'filter': f"{row['row_id']}_{row['room_id']}"}).to_pandas()['filter'].tolist())
processed_test_ids = set(processed_test_ds.select_columns(['row_id', 'room_id']).map(
    lambda row: {'filter': f"{row['row_id']}_{row['room_id']}"}).to_pandas()['filter'].tolist())

old_train_dataset = old_train_dataset.filter(
    lambda row: f"{row['scene_id']}_{row['room_id']}" in processed_train_ids).sort(
    ['scene_id', 'room_id', ]).select_columns(['vertices', 'whitelist']).map(
    lambda row: {'vertices': row['vertices'].tolist(), 'whitelist': row['whitelist'].tolist()})
old_val_dataset = old_val_dataset.filter(lambda row: f"{row['scene_id']}_{row['room_id']}" in processed_val_ids).sort(
    ['scene_id', 'room_id', ]).select_columns(['vertices', 'whitelist']).map(
    lambda row: {'vertices': row['vertices'].tolist(), 'whitelist': row['whitelist'].tolist()})
old_test_dataset = old_test_dataset.filter(
    lambda row: f"{row['scene_id']}_{row['room_id']}" in processed_test_ids).sort(
    ['scene_id', 'room_id', ]).select_columns(['vertices', 'whitelist']).map(
    lambda row: {'vertices': row['vertices'].tolist(), 'whitelist': row['whitelist'].tolist()})

assert processed_train_ds.count() == old_train_dataset.count()
assert processed_val_ds.count() == old_val_dataset.count()
assert processed_test_ds.count() == old_test_dataset.count()

processed_train_ds.zip(old_train_dataset).write_parquet(
    '/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_reprocessed/train/')
processed_val_ds.zip(old_val_dataset).write_parquet('/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_reprocessed/val/')
processed_test_ds.zip(old_test_dataset).write_parquet(
    '/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_reprocessed/test/')

print(f"train: {processed_train_ds.count()}, val: {processed_val_ds.count()}, test: {processed_test_ds.count()}")
