import os

from data.processing.procthor_process_scene import process_scene

if __name__ == "__main__":

    dataset_root = "/Volumes/Seagate/ai2thor/ai2thor-hab"
    procthor_scene_configs_dir = f"{dataset_root}/configs/scenes/ProcTHOR"
    scene_dirs = sorted(os.listdir(procthor_scene_configs_dir))
    output_dir = os.path.join("/Users/dhruvabharadwaj/StageVista/data", "ProcTHOR_processed")
    for scene_dir in scene_dirs:
        if scene_dir.startswith("._"):
            continue
        for scene_name in sorted(os.listdir(os.path.join(procthor_scene_configs_dir, scene_dir))):
            if scene_name.startswith("._"):
                continue
            split = "train" if "Train" in scene_name else "val" if "Val" in scene_name else "test"
            process_scene(scene_name.rstrip(".scene_instance.json"), scene_dir, dataset_root, split, output_dir)
