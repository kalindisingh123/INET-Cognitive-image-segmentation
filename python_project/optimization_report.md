# Optimization Findings for Skin Cancer Detection System

## 1. Hyperparameter Tuning
- **Batch Size:** A batch size of 16 to 32 is optimal for the ISIC dataset to balance memory usage and gradient stability.
- **Learning Rate:** Starting with a learning rate of `1e-4` for U-Net and `1e-3` for the ResNet50 top layers yields the fastest convergence.
- **Dropout:** A dropout rate of 0.5 in the classification head prevents overfitting on the relatively small HAM10000 dataset.

## 2. Optimizer Selection
- **Adam:** Generally superior for segmentation tasks due to its adaptive learning rate properties.
- **SGD with Momentum:** Often better for the final fine-tuning stage of classification models to achieve better generalization.

## 3. Learning Rate Scheduling & Checkpointing
- **ReduceLROnPlateau:** Highly effective. Reducing the learning rate by a factor of 0.2 when validation loss plateaus for 5 epochs helps the model settle into local minima more effectively.
- **Early Stopping:** Prevents overfitting by halting training when validation performance stops improving for 10 consecutive epochs.
- **Model Checkpointing:** Implemented using `ModelCheckpoint` to automatically save the best performing weights (`val_loss` based) during training. This ensures that the most optimal version of the model is preserved even if training is interrupted or overfits in later epochs.
- **Resume Capability:** The training script now includes logic to detect existing weight files (`.h5`) and load them before starting the training loop, allowing for seamless resumption of training sessions.

## 4. Transfer Learning & Fine-Tuning
- **Stage 1:** Freeze the ResNet50 backbone and train only the custom dense layers.
- **Stage 2:** Unfreeze the top layers of ResNet50 and train with a very low learning rate (`1e-5`) to adapt the pre-trained features to dermatological specifics.

## 5. Data Augmentation
- Techniques like **Random Rotation**, **Horizontal/Vertical Flips**, and **Color Jittering** are crucial for skin cancer images to account for different camera angles and lighting conditions.
