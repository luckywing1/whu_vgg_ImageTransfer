"""
模块1：数据加载
- 加载图像（PIL）
- 预处理（减去 ImageNet 均值）
- 反预处理（加回均值，clip，转 uint8）
"""
import numpy as np
from PIL import Image

# VGG19 在 ImageNet 上训练时使用的 BGR/RGB 像素均值
# imagenet-vgg-verydeep-19.mat 中的均值是 RGB 顺序，对应 [123.68, 116.78, 103.94]
MEAN_PIXEL = np.array([123.68, 116.779, 103.939], dtype=np.float32)


def load_image(path, max_size=512, shape=None):
    """读取图像并 resize
    参数:
        path: 图像路径
        max_size: 最长边像素，按比例缩放
        shape: (H, W) 元组；若指定则强制 resize 到该尺寸（用于让 style 与 content 对齐）
    返回:
        ndarray, shape=(1, H, W, 3), dtype=float32, 范围 [0, 255]
    """
    img = Image.open(path).convert('RGB')

    if shape is not None:
        # PIL.resize 接收 (W, H)
        img = img.resize((shape[1], shape[0]), Image.LANCZOS)
    else:
        w, h = img.size
        if max(w, h) > max_size:
            if w >= h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.float32)  # (H, W, 3)
    arr = arr[np.newaxis, ...]               # (1, H, W, 3)
    return arr


def preprocess(img):
    """减去 ImageNet 均值；输入 [0,255] float32"""
    return img - MEAN_PIXEL


def deprocess(img):
    """加回均值，clip 到 [0,255]，转 uint8；输入 shape=(1,H,W,3) 或 (H,W,3)"""
    if img.ndim == 4:
        img = img[0]
    img = img + MEAN_PIXEL
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
