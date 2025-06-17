import os

from data.processing.procthor_process_scene import process_scene

if __name__ == "__main__":

    dataset_root = "/Volumes/UnionSine/ai2thor/ai2thor-hab"
    procthor_scene_configs_dir = f"{dataset_root}/configs/scenes/ArchitecTHOR"
    scene_dirs = sorted(os.listdir(procthor_scene_configs_dir))
    output_dir = os.path.join("/Users/dhruvabharadwaj/StageVista/data", "ArchitecTHOR_processed")

    for scene_name in sorted(os.listdir(procthor_scene_configs_dir)):
        if scene_name.startswith("._"):
            continue
        process_scene(scene_name[:-20], dataset_root, output_dir)
