import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

def calculate_spread(mask):
    """
    Calculate the percentage of the tumor area relative to the total image area.
    """
    tumor_pixels = np.sum(mask > 0.5)
    total_pixels = mask.size
    spread_percentage = (tumor_pixels / total_pixels) * 100
    return spread_percentage

def get_gradcam(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
