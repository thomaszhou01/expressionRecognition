from scipy.misc import face
import tensorflow as tf
import numpy as np
from numpy import loadtxt
import pandas as pd
from tensorflow.python import keras
import keras.models
import keras.layers


faceMeshData = loadtxt('data/data.csv', delimiter=',')
actualExpressions = loadtxt('data/labels.csv', delimiter=',')
actualExpressions = np.reshape(actualExpressions, (-1, 1))
actualExpressions = actualExpressions.astype(int)
print(np.reshape(faceMeshData[0], (-1, 1404)))
print(actualExpressions[0])
train_dataset = tf.data.Dataset.from_tensor_slices((faceMeshData,actualExpressions))


BATCH_SIZE = 8
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

model = keras.models.Sequential([
    keras.layers.Dense(1404),
    keras.layers.Dense(1404, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(faceMeshData, actualExpressions, epochs=20)
test = model.evaluate(faceMeshData, actualExpressions, verbose=1) 
print(test)
print(model.predict(np.reshape(faceMeshData[0], (-1, 1404))))
model.save('models/jan23.h5')
