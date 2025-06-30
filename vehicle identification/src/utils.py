import os
import cv2
import random
import numpy as np

def load_images_from_subdirectories(directory):
    image_files = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_files.append(os.path.join(subdir, file))
    return image_files

# def load_and_sample_images(directory, sample_size=300):
#     image_files = [os.path.join(directory, f) for f in os.listdir(directory)
#                    if os.path.isfile(os.path.join(directory, f))]
#     if len(image_files) > sample_size:
#         return random.sample(image_files, sample_size)
#     return image_files

def load_and_sample_images(directory, sample_size=300):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, f))

    if len(image_files) >= sample_size:
        return random.sample(image_files, sample_size)
    return image_files  # return all if less than 300


def slide_window(img_shape, x_start_stop=[None, None], y_start_stop=[None, None],
                 window_size=(128, 128), stride=16):
    """
    Generates a list of window positions (top-left and bottom-right coordinates)
    for sliding window detection.

    Parameters:
    - img_shape: Shape of the image (height, width)
    - x_start_stop: [x_start, x_stop] range for x-axis
    - y_start_stop: [y_start, y_stop] range for y-axis
    - window_size: (width, height) of the sliding window
    - stride: number of pixels to move the window at each step

    Returns:
    - List of tuples: [(x1, y1, x2, y2), ...]
    """
    img_height, img_width = img_shape[:2]

    # Define start and stop if None
    x_start = x_start_stop[0] if x_start_stop[0] is not None else 0
    x_stop  = x_start_stop[1] if x_start_stop[1] is not None else img_width
    y_start = y_start_stop[0] if y_start_stop[0] is not None else 0
    y_stop  = y_start_stop[1] if y_start_stop[1] is not None else img_height

    window_list = []

    for y in range(y_start, y_stop - window_size[1] + 1, stride):
        for x in range(x_start, x_stop - window_size[0] + 1, stride):
            x1, y1 = x, y
            x2, y2 = x + window_size[0], y + window_size[1]
            window_list.append((x1, y1, x2, y2))

    return window_list


def ensure_bgr(img):
    """
    Take an image array of shape:
      - (H, W)           : grayscale float or uint8
      - (H, W, 1)        : single‐channel
      - (H, W, 3)        : RGB or BGR
    Returns a proper 3‐channel BGR uint8 image.
    """
    # If float in [0,1], convert to uint8
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
        
    # Grayscale (H, W)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Single‐channel (H, W, 1)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)
    
    # Already 3‐channel; assume it's RGB → convert to BGR
    if img.ndim == 3 and img.shape[2] == 3:
        # If it's actually RGB, swap to BGR; if it's BGR, it still works
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    raise ValueError(f"Unsupported image shape: {img.shape}")


def create_dir_if_not_exists(directory):
    """
    Creates the directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
