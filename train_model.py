import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('fer2013.csv')

pixels = data['pixels'].tolist()
faces = []

for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split()]
    face = np.asarray(face).reshape(48,48)
    faces.append(face)

faces = np.asarray(faces)
faces = faces / 255.0
faces = faces.reshape(-1,48,48,1)

labels = pd.get_dummies(data['emotion']).values

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(faces, labels, epochs=5, batch_size=64)

model.save("emotion_model.h5")