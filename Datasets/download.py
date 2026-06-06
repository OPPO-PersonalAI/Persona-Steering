"""
Download published evaluation artifacts from Hugging Face Hub.

Usage:
  huggingface-cli login   # or set HF_TOKEN
  python download.py
"""
from huggingface_hub import snapshot_download
import os

repo_id = "PersonalAILab/Persona-Steer"
folder_to_download = "lamp_eval_profile_steering_results"
local_dir = "./downloaded_results"

print(f"Downloading '{folder_to_download}' from '{repo_id}'...")

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    allow_patterns=f"{folder_to_download}/*",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print(f"Done. Files saved under: {os.path.abspath(local_dir)}")
