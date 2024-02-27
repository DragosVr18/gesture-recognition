import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

dataset = tf.keras.utils.image_dataset_from_directory(
    'hagrid-classification-512p',
    color_mode='grayscale',
    label_mode='int',
    image_size=(256,256), 
    crop_to_aspect_ratio=True)

train = dataset.take(4400)
test = dataset.skip(4400)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',name='conv_layer_input'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',name='conv_hidden_1'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(96, (3, 3), activation='relu',name='conv_hidden_2'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(96, activation='relu',name='dense_hidden'),
    tf.keras.layers.Dense(6, activation='softmax',name='dense_output'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(train, epochs=8)

model.evaluate(test, batch_size=32, verbose=1)

model.save('conv_model.keras')
