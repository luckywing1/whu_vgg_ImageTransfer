"""图像保存工具"""
import os
import numpy as np
from PIL import Image


def save_image(arr, path):
    """保存 uint8 ndarray 为图像"""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
