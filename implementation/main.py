"""
network implementation is similar to NVIDEA End-to-End Self Driving Car

transfer learning is used -- network is trained on small and different dataset at a time
this helps in keeping track what features(images) and labels(steering) are improving accuracy
it also helps to train on a relatively small but effective dataset.
after each training, model is saved and reused the next time.
"""
import sys

sys.path.append("implementation")
from helper import Helper

# train and validation samples
train_samples, validation_samples = Helper.load_samples()

# generator to get batches for train and validation
train_generator = Helper.generator(train_samples, batch_size=32)
validation_generator = Helper.generator(validation_samples, batch_size=32)

# load pre-trained (transfer learning) or retrain entire network
model = Helper.get_model()

# train the network
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=len(train_samples),
                              validation_data=validation_generator,
                              validation_steps=len(validation_samples),
                              verbose=2,
                              epochs=3)

# save model and show the validation accuracy history
Helper.save_model(model, history)
