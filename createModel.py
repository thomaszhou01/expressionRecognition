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


model = keras.models.Sequential([
    keras.layers.Dense(1404),
    keras.layers.Dense(1404, activation='relu'),
    keras.layers.Dense(4)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics=['accuracy'])
model.fit(faceMeshData, actualExpressions, batch_size=32, shuffle=True, epochs=20)
test = model.evaluate(faceMeshData, actualExpressions, verbose=1) 
print(test)
print(model.predict(np.reshape(faceMeshData[0], (-1, 1404))))
model.save('models/jan23.h5')
