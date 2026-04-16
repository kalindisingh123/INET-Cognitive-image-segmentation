import os
import numpy as np
import tensorflow as tf
from models.unet import unet_model
from models.classifier import classifier_model
from utils.preprocessing import preprocess_image

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Configuration
BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE = (256, 256)
UNET_WEIGHTS_PATH = 'unet_best_weights.h5'
CLASSIFIER_WEIGHTS_PATH = 'classifier_best_weights.h5'

def train_segmentation():
    print("Initializing U-Net for Segmentation...")
    model = unet_model(input_size=(*IMG_SIZE, 3))
    
    # Check for existing weights to resume training
    if os.path.exists(UNET_WEIGHTS_PATH):
        print(f"Loading existing weights from {UNET_WEIGHTS_PATH} to resume training...")
        model.load_weights(UNET_WEIGHTS_PATH)

    # Optimization: Using Adam with a custom learning rate and scheduling
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks for optimization and checkpointing
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        UNET_WEIGHTS_PATH, 
        monitor='val_loss', 
        save_best_only=True, 
        save_weights_only=True, 
        verbose=1
    )
    
    callbacks = [lr_scheduler, early_stopping, checkpoint]
    
    print("Segmentation model training code ready with Checkpointing and Resume capability.")
    # model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=callbacks)

def train_classification():
    print("Initializing ResNet50 for Classification...")
    model = classifier_model(input_shape=(*IMG_SIZE, 3))
    
    # Check for existing weights to resume training
    if os.path.exists(CLASSIFIER_WEIGHTS_PATH):
        print(f"Loading existing weights from {CLASSIFIER_WEIGHTS_PATH} to resume training...")
        model.load_weights(CLASSIFIER_WEIGHTS_PATH)

    # Optimization Stage 1: Training only the top layers
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks for optimization and checkpointing
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        CLASSIFIER_WEIGHTS_PATH, 
        monitor='val_loss', 
        save_best_only=True, 
        save_weights_only=True, 
        verbose=1
    )
    
    callbacks = [lr_scheduler, early_stopping, checkpoint]
    
    print("Classification model training code ready with Checkpointing and Resume capability.")
    # model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=callbacks)

if __name__ == "__main__":
    # train_segmentation()
    # train_classification()
    pass
