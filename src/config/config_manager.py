import os

import yaml


class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the ConfigManager with a path to a configuration file.
        If the file does not exist, it uses a default configuration.
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Loads configuration from the YAML file specified by config_path.
        Returns:
            A dictionary of configuration parameters.
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as file:
                    config = yaml.safe_load(file)
                    print(f"[ConfigManager] Loaded configuration from {self.config_path}.")
                    return config
            except Exception as e:
                print(f"[ConfigManager] Error loading configuration file: {e}")
                print("[ConfigManager] Falling back to default configuration.")
        else:
            print(f"[ConfigManager] {self.config_path} not found. Using default configuration.")
        return self.default_config()

    def default_config(self) -> dict:
        """
        Returns a default configuration dictionary.
        Customize these defaults as needed.
        """
        return {
            "num_furniture_classes": 33,
            "max_input_furniture_pieces": 23,
            "max_output_furniture_pieces": 23,
            "num_epochs": 300,
            "learning_rate": 2e-4,
            "batch_size": 18,
            "diffusion_steps": 1000,
            "diffusion_semantic_kwargs": {
                "att_1": 0.99999,
                "att_T": 0.000009,
                "ctt_1": 0.000009,
                "ctt_T": 0.99999,
                "model_output_type": "x0",
                "mask_weight": 1,
                "auxiliary_loss_weight": 0.0005,
                "adaptive_auxiliary_loss": True
            },
            "diffusion_geometric_kwargs": {
                "schedule_type": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "loss_type": "mse",
                "model_mean_type": "eps",
                "model_var_type": "fixedsmall"
            },
            "data_path": {
                "train": "/Volumes/UnionSine/ProcTHOR_final_processed/train/",
                "val": "/Volumes/UnionSine/ProcTHOR_final_processed/val/",
                "test": "/Volumes/UnionSine/ProcTHOR_final_processed/test/"
            },
            "model": {
                "name": "placer",
                "hidden_size": 512,
                "dropout": 0.1,
            },
            "logger": {
                "project_name": "placer"
            },
            "ray": {
                "num_workers": 4,
                "use_gpu": True
            }
        }

    def get_config(self) -> dict:
        """
        Returns the loaded configuration.
        """
        return self.config
