import os
from huggingface_hub import snapshot_download

HF_REPO_ID = os.getenv("HF_REPO_ID", "eslamaly/fake-face-detector")  # this is your Space repo
HF_SUBDIR  = os.getenv("HF_SUBDIR", "fake_face_model")              # folder inside the repo
HF_TOKEN   = os.getenv("HF_TOKEN")                                  # only if private

def ensure_model_dir() -> str:
    """
    Downloads the Space repo files needed for the SavedModel folder.
    Returns local path to the SavedModel directory.
    """
    local_root = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="space",
        token=HF_TOKEN,
        # Only download the model folder (saves bandwidth/time)
        allow_patterns=[f"{HF_SUBDIR}/**"],
        local_dir="models_cache",
    )

    model_dir = os.path.join(local_root, HF_SUBDIR)
    if not os.path.exists(model_dir):
        raise RuntimeError(f"Model directory not found after download: {model_dir}")

    return model_dir