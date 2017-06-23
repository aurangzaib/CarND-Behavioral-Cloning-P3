from keras.layers import Conv2D, Lambda, Dropout, Dense, Flatten, Cropping2D
from sklearn.model_selection import train_test_split
from helper import load_samples, generator, show_history
from keras.models import Sequential
import os

cwd = os.getcwd()

folder = '/data-set-3-optimized'
csv_file = folder + '/driving_log.csv'
img_file = cwd + folder + '/IMG/'
train_samples, validation_samples = load_samples(csv_file, 3)
train_generator = generator(img_file, train_samples, batch_size=32)
validation_generator = generator(img_file, validation_samples, batch_size=32)
shape = (160, 320, 3)

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# flatten
model.add(Flatten())
model.add(Dense(units=100))
model.add(Dense(units=50))
model.add(Dense(units=10))
model.add(Dense(units=1))
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator,
                              steps_per_epoch=len(train_samples),
                              validation_data=validation_generator,
                              validation_steps=len(validation_samples),
                              verbose=2,
                              epochs=3)
model.save('model.h5')
print("model: {}".format(model.summary()))
show_history(history.history)
