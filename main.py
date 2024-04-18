import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (150, 150))
            images.append(img)
    return images

def preprocess_dataset(dataset_path):
    brown_spot_images = load_images_from_folder(os.path.join(dataset_path, "brown_spot"))
    leaf_blight_images = load_images_from_folder(os.path.join(dataset_path, "leaf_blight"))
    false_smut_images = load_images_from_folder(os.path.join(dataset_path, "false_smut"))

    images = np.array(brown_spot_images + leaf_blight_images + false_smut_images)
    labels = np.array([0]*len(brown_spot_images) + [1]*len(leaf_blight_images) + [2]*len(false_smut_images)) # 0 for brown spot, 1 for leaf blight, 2 for false smut

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_dataset('paddy_leaves_dataset')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax') # 3 classes: brown spot, leaf blight, and false smut
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model = create_model()
