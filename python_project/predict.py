import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.unet import unet_model
from models.classifier import classifier_model
from utils.preprocessing import preprocess_image
from utils.analysis import calculate_spread, get_gradcam

def run_prediction(image_path):
    # 1. Preprocess
    img = preprocess_image(image_path)
    img_input = np.expand_dims(img, axis=0)

    # 2. Load Models (Assuming weights are saved)
    # seg_model = unet_model()
    # seg_model.load_weights('unet_weights.h5')
    # class_model = classifier_model()
    # class_model.load_weights('resnet_weights.h5')

    # 3. Predict
    # mask = seg_model.predict(img_input)[0]
    # label_probs = class_model.predict(img_input)[0]
    
    # Mock results for demonstration
    mask = np.zeros((256, 256))
    mask[100:150, 100:150] = 1.0 # Simulated tumor
    label = "Malignant" if np.random.rand() > 0.5 else "Benign"
    
    # 4. Analysis
    spread = calculate_spread(mask)
    
    # 5. Visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img)
    
    plt.subplot(1, 3, 2)
    plt.title(f"Segmented Mask (Spread: {spread:.2f}%)")
    plt.imshow(mask, cmap='jet')
    
    plt.subplot(1, 3, 3)
    plt.title(f"Prediction: {label}")
    plt.imshow(img)
    plt.imshow(mask, alpha=0.3, cmap='Reds') # Overlay
    
    plt.show()

if __name__ == "__main__":
    # run_prediction('test_image.jpg')
    pass
