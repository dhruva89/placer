import os
import tempfile
from collections import defaultdict

import numpy as np
import ray
import ray.train
import torch
import torch.optim as optim
import yaml
from gradnorm_pytorch import GradNormLossWeighter
from ray.train import Checkpoint, DataConfig, ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

from src.config.config_manager import ConfigManager
from src.model.networks.mixed_diffusion import MixedDiffusion


def train_func(config):
    """Your training function that will be launched on each worker."""
    from torch.nn.attention import sdpa_kernel
    from torch.nn.attention import SDPBackend
    import neptune

    device = ray.train.torch.get_device()
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

    # Prepare Ray Data Loaders
    # ====================================================
    train_ds = ray.train.get_dataset_shard("train")
    eval_ds = ray.train.get_dataset_shard("validation")

    if ray.train.get_context().get_world_rank() == 0:

        run = neptune.init_run(
            project=config["project"],
            api_token=config["api_token"],
            name=config["exp_name"],
        )
    else:
        run = None

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

    train_dataloader = train_ds.iter_torch_batches(
        batch_size=config['batch_size'], collate_fn=collate_fn
    )
    eval_dataloader = eval_ds.iter_torch_batches(
        batch_size=config['batch_size'], collate_fn=collate_fn
    )
    # ====================================================

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    loss_weighter = GradNormLossWeighter(
        num_losses=5,
        loss_names=tuple(["loss.semantic",
                          "loss.semantic_aux",
                          "loss.geometric_position",
                          "loss.geometric_orientation",
                          "loss.pointer"]),
        learning_rate=config['learning_rate'],
        restoring_force_alpha=0.,
        grad_norm_parameters=None,
    )
    loss_weighter._grad_norm_parameters = tuple(model.denoiser.parameters())

    for epoch in range(1, config['num_epochs'] + 1):

        # — accumulators for this epoch’s train sublosses
        train_sums = defaultdict(float)
        train_batches = 0

        # — optional: accumulators for val sublosses
        val_sums = defaultdict(float)
        val_batches = 0

        # TRAIN loop
        for batch in train_dataloader:
            model.train()
            verts = batch["vertices"]
            furn = batch["furniture_points_in"]
            sem = batch["semantic_class_in"]
            idx = batch["input_idx"]
            gt_geom = batch["geometric"]
            gt_sem = batch["semantic_class"]
            input_mask = batch["input_mask"]
            output_mask = batch["output_mask"]

            with sdpa_kernel(SDPBackend.MATH):
                loss_dict = model(verts, furn, sem, gt_sem, gt_geom, idx, input_mask, output_mask)

            optimizer.zero_grad()
            with sdpa_kernel(SDPBackend.MATH):
                loss_weighter.backward(loss_dict, retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # accumulate each subloss
            for key, sub in loss_dict.items():
                train_sums[key] += sub if isinstance(sub, (int, float)) else sub.item()

            train_batches += 1

        # (Optional) one pass through validation set for epoch-level val sublosses
        with torch.no_grad():
            model.eval()
            for val_batch in eval_dataloader:
                v_verts = val_batch["vertices"]
                v_furn = val_batch["furniture_points_in"]
                v_sem = val_batch["semantic_class_in"]
                v_idx = val_batch["input_idx"]
                v_gt_geom = val_batch["geometric"]
                v_gt_sem = val_batch["semantic_class"]
                v_input_mask = val_batch["input_mask"]
                v_output_mask = val_batch["output_mask"]

                with sdpa_kernel(SDPBackend.MATH):
                    v_loss_dict = model(v_verts, v_furn, v_sem, v_gt_sem, v_gt_geom, v_idx, v_input_mask,
                                        v_output_mask)

                for key, sub in v_loss_dict.items():
                    val_sums[key] += sub if isinstance(sub, (int, float)) else sub.item()
                val_batches += 1

        metrics = {}
        # compute and log epoch averages
        train_total = 0.0
        for key, total in train_sums.items():
            avg = total / max(train_batches, 1)
            train_total += avg
            metrics[f"train/{key}"] = avg

        val_total = 0.0
        if val_batches > 0:
            for key, total in val_sums.items():
                avg = total / val_batches
                val_total += avg
                metrics[f"val/{key}"] = avg

        if run is not None:
            for k, v in metrics.items():
                run[k].append(v)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None

            should_checkpoint = epoch % 20 == 0
            # In standard DDP training, where the model is the same across all ranks,
            # only the global rank 0 worker needs to save and report the checkpoint
            if ray.train.get_context().get_world_rank() == 0 and should_checkpoint:
                torch.save(
                    model.state_dict(),  # NOTE: Unwrap the model.
                    os.path.join(temp_checkpoint_dir, "model.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            ray.train.report(metrics, checkpoint=checkpoint)

    if run is not None:
        run.stop()


if __name__ == "__main__":
    num_workers = 8
    use_gpu = True

    training_config = ConfigManager().get_config()

    with open("secrets.yaml", "r") as stream:
        api_key = yaml.safe_load(stream)['neptune_api_key']

    training_config["exp_name"] = 'regular-attention_no-aabb'
    training_config["api_token"] = api_key
    training_config["project"] = 'stage-vista/placer'

    # Configure computation resources
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    # Prepare Ray Datasets
    ray_datasets = {
        "train": ray.data.read_parquet('/home/ubuntu/temp/ProcTHOR_normalized_sorted_filtered/train'),
        "validation": ray.data.read_parquet('/home/ubuntu/temp/ProcTHOR_normalized_sorted_filtered/val'),
    }

    trainer = TorchTrainer(
        train_func,
        train_loop_config=training_config,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        datasets=ray_datasets,
        dataset_config=DataConfig(datasets_to_split=["train", "validation"]),
        run_config=RunConfig()
    )

    result = trainer.fit()
