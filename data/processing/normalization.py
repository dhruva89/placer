import numpy as np
import ray

train_dataset = ray.data.read_parquet('/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_final_processed/train/')
val_dataset = ray.data.read_parquet('/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_final_processed/val/')
test_dataset = ray.data.read_parquet('/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_final_processed/test/')

datasets = [(train_dataset, 'train'), (val_dataset, 'val'), (test_dataset, 'test')]

max_furniture_pieces = 30


def gather_points(row):
    vertices = np.asarray(row['vertices'])
    position = np.asarray(row['position'])
    orientation = np.asarray(row['rotation'])
    size = np.asarray(row['size'])
    furniture_vertices = np.asarray(row['furniture_vertices'])

    centroid = vertices.mean(axis=0, keepdims=True)

    # Center the points.
    vertices = vertices - centroid
    furniture_vertices = furniture_vertices - centroid
    position = position - centroid

    # Compute the maximum norm for scaling.
    norms = np.linalg.norm(vertices, axis=1, ord=2).max(axis=0, keepdims=True)

    # Scale the points and AABB.
    vertices /= norms
    furniture_vertices /= norms
    position /= norms
    size /= norms

    row['vertices'] = vertices
    row['furniture_vertices'] = furniture_vertices
    row['position'] = position
    row['size'] = size
    row['rotation'] = orientation
    row['centroid'] = centroid
    row['norms'] = norms
    del row['size']

    return row


for (dataset, split) in datasets:
    dataset.map(gather_points).write_parquet(
        f'/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_normalized/{split}')
