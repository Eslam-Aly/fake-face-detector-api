import os
from huggingface_hub import snapshot_download

HF_REPO_ID = os.getenv("HF_REPO_ID", "eslamaly/fake-face-detector")  # Space repo id
HF_SUBDIR  = os.getenv("HF_SUBDIR", "fake_face_model")              # folder inside Space
HF_TOKEN   = os.getenv("HF_TOKEN")                                  # only if private

def ensure_model_dir() -> str:
    local_root = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="space",
        token=HF_TOKEN,
        allow_patterns=[f"{HF_SUBDIR}/**"],
        local_dir="models_cache",
    )

    model_dir = os.path.join(local_root, HF_SUBDIR)
    if not os.path.exists(model_dir):
        raise RuntimeError(f"Model directory not found: {model_dir}")

    return model_dir