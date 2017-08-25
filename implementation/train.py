"""
network implementation is similar to NVIDEA End-to-End Self Driving Car

transfer learning is used -- network is trained on small data and different at a time
this helps in keeping track what features(images) and labels(steering) are improving accuracy
it also helps to train on a relatively small but effective dataset.
after each training, model is saved and reused the next time.
"""
import os
import sys

sys.path.append("implementation/")
from keras.models import load_model

from configuration import Configuration
from helper import load_samples, generator, show_history
from classifier import Classifier

config = Configuration().__dict__
# directories
cwd = os.getcwd()
folder = '/data'
csv_file = cwd + folder + '/driving_log.csv'
img_file = cwd + folder + '/IMG/'

# train and validation samples
train_samples, validation_samples = load_samples(csv_file)

# generator to get batches for train and validation
train_generator = generator(img_file, train_samples, batch_size=32)
validation_generator = generator(img_file, validation_samples, batch_size=32)

# load pre-trained (transfer learning) or retrain entire network
use_pre_trained = False
model = load_model('../model-pre-trained.h5') if use_pre_trained else Classifier.implement_classifier(config["shape"])

# # print layers of the models
# for layer in model.layers:
#     print(layer.get_weights())

print(model.summary())
print(model.count_params())

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=len(train_samples),
                              validation_data=validation_generator,
                              validation_steps=len(validation_samples),
                              verbose=2,
                              epochs=3)
model.save('model.h5')
show_history(history.history)
