import numpy as np
import ray
import torch
import yaml

from src.config.config_manager import ConfigManager


class FurniturePlacementModelBase:

    def __init__(self):
        # Load configuration
        self.config = ConfigManager().get_config()
        with open('/Users/dhruvabharadwaj/StageVista/repos/placer/secrets.yaml', 'r') as f:
            secrets = yaml.safe_load(f)
        self.neptune_api_key = secrets['neptune_api_key']

        # Load training dataset using Ray Data
        self.train_dataset = ray.data.read_parquet(
            '/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_normalized/train/')

        # Load val dataset using Ray Data
        self.val_dataset = ray.data.read_parquet('/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_normalized/val/')

        # Load val dataset using Ray Data
        self.test_dataset = ray.data.read_parquet(
            '/Users/dhruvabharadwaj/StageVista/data/ProcTHOR_normalized/test/')

    def collate_fn(self, batch):
        device = ray.train.torch.get_device()
        max_furniture_pieces = self.config.get("max_furniture_pieces")
        num_furniture_classes = self.config.get("num_furniture_classes")

        padded_input_furniture_class = np.array([
            np.pad(arr, (0, max_furniture_pieces - len(arr)), 'constant', constant_values=num_furniture_classes)
            for arr in batch['semantic_class']
        ])
        padded_input_furniture_class = torch.tensor(padded_input_furniture_class, dtype=torch.int64, device=device)

        furniture_pc_list = []
        for arr in batch['furniture_vertices']:
            arr = np.asarray(arr, dtype=np.float32)  # (N, 1024, 3)
            n = arr.shape[0]
            if n < max_furniture_pieces:
                pad_n = max_furniture_pieces - n
                # create `pad_n` random clouds of shape (1024, 3)
                noise = np.random.rand(pad_n, arr.shape[1], arr.shape[2]).astype(np.float32)
                arr = np.concatenate([arr, noise], axis=0)  # now (max_furniture_pieces, 1024, 3)
            else:
                arr = arr[:max_furniture_pieces]
            furniture_pc_list.append(arr)

        # stack into (batch_size, max_furniture_pieces, 1024, 3)
        padded_input_furniture_pc = torch.tensor(
            np.stack(furniture_pc_list),
            dtype=torch.float32,  # point clouds are floats, not ints
            device=device
        )

        padded_input_furniture_idx = np.array([
            np.pad(arr, (0, max_furniture_pieces - len(arr)), 'constant', constant_values=max_furniture_pieces)
            for arr in batch['input_idx']
        ])
        padded_input_furniture_idx = torch.tensor(padded_input_furniture_idx, dtype=torch.int64, device=device)
        mask = (padded_input_furniture_idx != max_furniture_pieces).long()

        # Process and concatenate geometric features
        geometric = []
        for poss, orients in zip(batch['position'], batch['rotation']):
            item_list = []
            for pos, orient in zip(poss, orients):
                concatenated = np.concatenate(
                    [pos, orient])
                item_list.append(concatenated)
            if not item_list:
                print(item_list)
            geometric.append(np.stack(item_list))

        # Pad geometric arrays along the first dimension (number of pieces)
        padded_geometric = np.stack([
            np.pad(arr, ((0, max_furniture_pieces - len(arr)), (0, 0)), mode='constant', constant_values=0.0)
            for arr in geometric
        ], dtype=np.float32)

        vertices = np.stack([
            np.stack([np.array(val, dtype=np.float32) for val in vertices_row])
            for vertices_row in batch['vertices']
        ])

        return {
            'room_id': batch['room_id'],
            'scene_id': batch['scene_id'],
            'vertices': torch.tensor(vertices, dtype=torch.float32, device=device).permute(0, 2, 1),
            'furniture_vertices': padded_input_furniture_pc,
            'semantic_class': padded_input_furniture_class,
            'input_idx': padded_input_furniture_idx,
            'geometric': torch.tensor(padded_geometric, dtype=torch.float32, device=device),
            'mask': mask
        }
