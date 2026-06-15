import os
import yaml
from typing import Any, Dict
from dotenv import load_dotenv


def load_config(config_path: str) -> Dict[str, Any]:
    load_dotenv()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["data_root"] = os.getenv("DATA_ROOT", cfg.get("data_root", "./data"))
    cfg["ckpt_dir"] = os.getenv("CKPT_DIR", cfg.get("ckpt_dir", "./checkpoints"))
    cfg["results_dir"] = os.getenv("RESULTS_DIR", cfg.get("results_dir", "./results"))
    cfg["log_dir"] = os.getenv("LOG_DIR", cfg.get("log_dir", "./logs"))
    cfg["seed"] = int(os.getenv("SEED", cfg.get("seed", 42)))
    return cfg


def save_config(cfg: Dict[str, Any], config_path: str) -> None:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
