'''Build an Artificial Neural Network by implementing the Back
propagation algorithm and test the same using appropriate data
sets.'''
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import numpy as np

model = Sequential([Input(shape=(2,)), Dense(4, activation='relu'), Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
model.fit(x, y, epochs=10)

test = np.array([[0,0],[0,1],[1,1],[1,1]])
print(model.predict(test))

plot_model(model, show_shapes=True, show_layer_names=True)
