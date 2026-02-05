import numpy as np
from PIL import Image
from io import BytesIO

def load_and_preprocess_image(file_bytes: bytes, image_size: int = 299) -> np.ndarray:
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    img = img.resize((image_size, image_size))

    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)