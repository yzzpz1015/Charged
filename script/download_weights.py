import os
from huggingface_hub import snapshot_download

SAVE_DIR = "/home/data/yzzpz/Charged/api/model/weights"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("="*60)
print(f"文件将保存至: {SAVE_DIR}")
print("="*60)

# 1. GPT-2
print("\n[1/2] 正在下载 GPT-2 (约 500MB)...")
snapshot_download(
    repo_id="gpt2",
    local_dir=f"{SAVE_DIR}/gpt2_model",
    local_dir_use_symlinks=False,
    resume_download=True
)
print("GPT-2 download done!")

# 2. SigLIP v2
print("\n[2/2] downloading SigLIP v2 (google/siglip2-base-patch16-224)...")
snapshot_download(
    repo_id="google/siglip2-base-patch16-224",
    local_dir=f"{SAVE_DIR}/siglip2_model",
    local_dir_use_symlinks=False,
    resume_download=True
)
print("SigLIP v2 download done!")

print(f"\nAll models saved to: {SAVE_DIR}")
