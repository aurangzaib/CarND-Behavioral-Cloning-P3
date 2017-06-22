"""
1- load features(images) and labels(steering commands)
2- apply regression model
3- save model as h5 file
"""

from helper import load_data
import tensorflow as tf

tf.python.control_flow_ops = tf

from keras.layers.core import Flatten, Dense
from keras.models import Sequential

x_train, y_train, shape = load_data('/data/driving_log.csv')
# regression network
model = Sequential()
model.add(Flatten(input_shape=shape))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train,
          epochs=10,
          batch_size=1024,
          validation_split=0.3,
          shuffle=True,
          verbose=False)
model.save('model.h5')
