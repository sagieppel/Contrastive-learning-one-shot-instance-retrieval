import cv2
import numpy as np


def resize_and_center_crop(img, size=512):
    """
    Resize image so smallest dimension is 512px, then center crop to 512x512.

    Args:
        img (np.ndarray): Input image as numpy array
        size (int): Target size for output square image (default: 512)

    Returns:
        np.ndarray: Processed 512x512 image
    """
    h, w = img.shape[:2]

    # Resize so smallest dimension equals target size
    if w < h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size

    img = cv2.resize(img, (new_w, new_h))

    # Center crop to square
    y = (new_h - size) // 2
    x = (new_w - size) // 2

    return img[y:y + size, x:x + size]