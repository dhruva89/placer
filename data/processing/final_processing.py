import json
import os

import numpy as np
import pandas as pd
import prior
import ray
from plyfile import PlyData

from data.processing.farthest_point_sampling import farthest_point_sampling

with open('/Users/dhruvabharadwaj/StageVista/repos/placer/data/processing/disallowed_room_objects.jsonl') as f:
    disallowed_room_objects = json.load(f)


def assign_room_id_and_scene_id(batch: pd.DataFrame) -> pd.DataFrame:
    batch['room_id'] = batch['path'].transform(lambda path: path.split("/")[-1].rstrip(".json"))
    batch['scene_id'] = batch['path'].transform(lambda path: path.split("/")[-2])
    return batch


def assign_room_type(row):
    room_id = row['room_id']
    scene_id = row['scene_id']

    if "Train" in scene_id:
        dataset = prior_dataset.train
    elif "Test" in scene_id:
        dataset = prior_dataset.test
    else:
        dataset = prior_dataset.val

    pos = int(scene_id.split("-")[-1])
    property_data = json.loads(dataset.select([pos]).data[0].decode('utf-8'))
    for room in property_data['rooms']:
        if room['id'] == room_id:
            row['room_type'] = room['roomType']
            break

    return row


def aggregate_room_objects(group: pd.DataFrame) -> pd.DataFrame:
    room_id = group['room_id'].iloc[0]
    room_type = group['room_type'].iloc[0]
    scene_id = group['scene_id'].iloc[0]
    semantic_classes = group['semantic_class'].tolist()
    disallowed_items = set(disallowed_room_objects[room_type])

    disallowed_indices = set()
    disallowed_input_idx = set()
    for disallowed_idx, semantic_class in enumerate(semantic_classes):
        if semantic_class in disallowed_items:
            disallowed_indices.add(disallowed_idx)
            disallowed_input_idx.add(group['input_idx'].iloc[disallowed_idx])

    mask = np.ones(len(group), dtype=bool)
    mask[list(disallowed_indices)] = False

    allowed_semantic_classes = group.loc[mask, 'semantic_class'].tolist()
    allowed_input_idx = group.loc[mask, 'input_idx']
    allowed_positions = group.loc[mask, 'position'].tolist()
    allowed_rotation = group.loc[mask, 'rotation'].tolist()
    allowed_size = group.loc[mask, 'size'].tolist()

    kept = group['input_idx'][~group['input_idx'].isin(disallowed_input_idx)]
    mapping = {old: new for new, old in enumerate(sorted(kept.unique()))}

    # Read the PLY files
    dirname = os.path.dirname(group['path'].iloc[0])
    ply_file_path = os.path.join(dirname, room_id + "_pointcloud.ply")
    ply = PlyData.read(ply_file_path)  # replace with your file path
    allowed_ids = sorted(kept.unique())
    furniture_ply_file_paths = [
        os.path.join(dirname, f"{room_id}_furniture|{x}_pointcloud.ply")
        for x in allowed_ids
    ]
    furniture_plys = [PlyData.read(furniture_ply_file_path) for furniture_ply_file_path in furniture_ply_file_paths]

    allowed_input_idx = allowed_input_idx.map(lambda x: mapping[x]).tolist()

    furniture_vertex_data = [ply['vertex'].data for ply in furniture_plys]

    if len(furniture_vertex_data) == 0:
        return pd.DataFrame([])

    furniture_vertices = np.stack(
        [np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T for vertex_data in furniture_vertex_data],
        axis=0)

    # Extract vertex data; this returns a numpy structured array
    vertex_data = ply['vertex'].data

    # Optionally, convert the structured array into a regular numpy array
    # by stacking the 'x', 'y', and 'z' fields (if they exist)
    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
    # Downsample the vertices to 1024 points using farthest point sampling
    vertices = farthest_point_sampling(vertices, 1024)

    set_type = "train" if "Train" in scene_id else "test" if "Test" in scene_id else "val"
    group.drop(['path'], axis=1, inplace=True)

    assert len(allowed_semantic_classes) == len(allowed_input_idx) == len(allowed_positions) == len(
        allowed_rotation) == len(allowed_size)
    assert set(allowed_input_idx) == set(range(furniture_vertices.shape[0]))

    df = pd.DataFrame([{
        'room_id': room_id,
        'scene_id': scene_id,
        'vertices': vertices.tolist(),
        'semantic_class': allowed_semantic_classes,
        'input_idx': allowed_input_idx,
        'position': allowed_positions,
        'size': allowed_size,
        'rotation': allowed_rotation,
        'split': set_type,
        'furniture_vertices': furniture_vertices.tolist()
    }])

    return df


def accumulate_row(acc, row):
    acc[row['semantic_class']] += 1
    return acc


in_dir = "/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_processed"
sub_dirs = [sub_dir for sub_dir in sorted(os.listdir(in_dir)) if not sub_dir.startswith(".")]

NUM_FURNITURE_CLASSES = 33

jsons = []

prior_dataset = prior.load_dataset('procthor-10k')

for sub_dir in sub_dirs:

    scene_dirs = [scene_dir for scene_dir in sorted(os.listdir(os.path.join(in_dir, sub_dir))) if
                  not scene_dir.startswith(".")]

    for scene_dir in scene_dirs:
        json_file_names = [file for file in os.listdir(os.path.join(in_dir, sub_dir, scene_dir)) if
                           file.endswith(".json") and not file.startswith("._")]

        for json_file_name in json_file_names:
            json_file = os.path.join(in_dir, sub_dir, scene_dir, json_file_name)
            jsons.append(json_file)

jsons = [json for json in jsons if os.path.getsize(json) > 2]

rooms_data = ray.data.read_json(
    jsons,
    include_paths=True,
    parallelism=4
)

rooms_data = rooms_data.map_batches(assign_room_id_and_scene_id, batch_format="pandas")
rooms_data = rooms_data.map(assign_room_type)

final_dataset = rooms_data.groupby('path').map_groups(aggregate_room_objects).filter(
    lambda row: len(row['semantic_class']) != 0).materialize()
train_dataset = final_dataset.filter(expr="split == 'train'").random_shuffle()
val_dataset = final_dataset.filter(expr="split == 'val'").random_shuffle()
test_dataset = final_dataset.filter(expr="split == 'test'").random_shuffle()
train_dataset.write_parquet('/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_final_processed/train/')
val_dataset.write_parquet('/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_final_processed/val/')
test_dataset.write_parquet('/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_final_processed/test/')
