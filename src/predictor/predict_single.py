import numpy as np
import ray
import torch

from src.config.config_manager import ConfigManager
from src.model.networks.mixed_diffusion import MixedDiffusion

config = ConfigManager().get_config()
device = 'mps'
model = MixedDiffusion(
    device,
    config['batch_size'],
    config.get("diffusion_semantic_kwargs"),
    config.get("diffusion_geometric_kwargs"),
    diffusion_steps=config['diffusion_steps'],
    num_furniture_classes=config.get("num_furniture_classes"),
    max_input_slots=config.get("max_input_furniture_pieces"),
    max_output_slots=config.get("max_output_furniture_pieces"),
).to(device)

model.load_state_dict(torch.load(
    '/Users/dhruvabharadwaj/Downloads/model.pt',
    weights_only=True, map_location=device))

model.eval()

def collate_fn(batch):
    max_input_furniture_pieces = config.get("max_input_furniture_pieces")
    max_output_furniture_pieces = config.get("max_output_furniture_pieces")
    num_furniture_classes = config.get("num_furniture_classes")

    padded_input_furniture_class = np.array([
        np.pad(arr, (0, max_input_furniture_pieces - len(arr)), 'constant', constant_values=num_furniture_classes)
        for arr in batch['semantic_class']
    ])
    padded_input_furniture_class = torch.tensor(padded_input_furniture_class, dtype=torch.int64, device=device)
    sem_mask = (padded_input_furniture_class != num_furniture_classes).long()

    furniture_pc_list = []
    for arr in batch['furniture_vertices']:
        arr = np.asarray(arr, dtype=np.float32)  # shape: (n_pieces, 1024, 3)
        n = arr.shape[0]
        if n < max_input_furniture_pieces:
            pad_n = max_input_furniture_pieces - n
            zeros = np.zeros((pad_n, arr.shape[1], arr.shape[2]), dtype=np.float32)
            arr = np.concatenate([arr, zeros], axis=0)
        else:
            arr = arr[:max_input_furniture_pieces]
        furniture_pc_list.append(arr)

    padded_input_furniture_pc = torch.tensor(
        np.stack(furniture_pc_list),
        dtype=torch.float32,
        device=device
    )  # (batch_size, max_pieces, 1024, 3)s

    padded_input_furniture_idx = np.array([
        np.pad(arr, (0, max_input_furniture_pieces - len(arr)), 'constant',
               constant_values=max_input_furniture_pieces)
        for arr in batch['input_idxs']
    ])
    padded_input_furniture_idx = torch.tensor(padded_input_furniture_idx, dtype=torch.int64, device=device)
    mask = (padded_input_furniture_idx != max_input_furniture_pieces).long()

    clamped_idx = padded_input_furniture_idx.clamp(max=max_input_furniture_pieces - 1)

    # 1) gather all (will be “wrong” in masked positions, but we’ll overwrite them)
    g = torch.gather(
        padded_input_furniture_class,  # shape [B, m]
        dim=1,
        index=clamped_idx  # ∈ [0…m−1]
    )

    # 2) wherever mask == 0, set to pad‐class = num_furniture_classes
    pad_value = num_furniture_classes  # this is your “semantic PAD” index
    gt_semantic = torch.where(
        mask.bool(),  # True→keep g, False→use pad_value
        g,
        torch.full_like(g, pad_value)
    )

    # Process and concatenate geometric features
    geometric = []
    for poss, orients in zip(batch['positions'], batch['rotations']):
        item_list = []
        for pos, orient in zip(poss, orients):
            concatenated = np.concatenate(
                [pos, np.deg2rad(orient)])
            item_list.append(concatenated)
        if not item_list:
            print(item_list)
        geometric.append(np.stack(item_list))

    # Pad geometric arrays along the first dimension (number of pieces)
    padded_geometric = np.stack([
        np.pad(arr, ((0, max_output_furniture_pieces - len(arr)), (0, 0)), mode='constant', constant_values=0.0)
        for arr in geometric
    ], dtype=np.float32)

    vertices = np.stack([
        np.stack([np.array(val, dtype=np.float32) for val in vertices_row])
        for vertices_row in batch['vertices']
    ])

    return {
        'vertices': torch.tensor(vertices, dtype=torch.float32, device=device).permute(0, 2, 1),
        'furniture_points_in': padded_input_furniture_pc.permute(0, 1, 3, 2),
        'semantic_class_in': padded_input_furniture_class,
        'semantic_class': gt_semantic,
        'input_idx': padded_input_furniture_idx,
        'geometric': torch.tensor(padded_geometric, dtype=torch.float32, device=device),
        'input_mask': sem_mask,
        'output_mask': mask
    }


iterable = iter(
    ray.data.read_parquet(
        '/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_normalized_aabb_filtered/val').iter_torch_batches(
        batch_size=1, collate_fn=collate_fn))

data = next(iterable)
_, geom, input_idx = model.sample(data['vertices'], data['furniture_points_in'], data['semantic_class'], 1)
position, orientation = torch.split(geom, [3, 3], dim=2)

print({"input_idx_pred": input_idx.cpu().tolist(), "position_pred": position.cpu().tolist(),
       "orientation_pred": orientation.cpu().tolist(), **{k: v.cpu().tolist() for k, v in data.items()}})
