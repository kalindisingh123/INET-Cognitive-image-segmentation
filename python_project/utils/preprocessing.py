import cv2
import numpy as np

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Load, resize, and normalize image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img

def augment_data(image):
    """
    Simple augmentation example.
    """
    # Flip
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
    return image
