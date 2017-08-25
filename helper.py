import csv
import os
import pickle
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Lambda, Dropout, Dense, Flatten, Cropping2D
from keras.models import Sequential
from matplotlib.pyplot import imread as imr
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_samples(csv_file, quantity=-1):
    """
    return lines from csv file
    each line contains left, center, right images and steering info
    @:param csv_file: csv file path
    @:param quantity: how many lines to be returned from csv file
    """

    samples = []
    # save samples
    with open(csv_file) as file:
        for line in csv.reader(file):
            samples.append(line)
    # shuffle and return samples
    samples = shuffle(samples)
    samples = samples if quantity is -1 else samples[:quantity]
    return train_test_split(samples, test_size=0.2)


def generator(_dir, samples, batch_size=32):
    """
    generator --> 1- infinite while loop 2- yield instead of return
    :param _dir: directory of the images
    :param samples: lines from csv file
    :param batch_size: how many features and labels each time
    :return: features, labels
    """

    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        # i think we don't need redundant shuffling
        # samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images, measurements = [], []
            for batch_sample in batch_samples:
                # steering
                steer, corr = float(batch_sample[3]), 0.2
                # center, left and right paths
                center = batch_sample[0].split('/')[-1]
                left = batch_sample[1].split('/')[-1]
                right = batch_sample[2].split('/')[-1]
                # check for image file existence for given path in csv
                if exists(_dir + center) and exists(_dir + left) and exists(_dir + right):
                    # center, left and right images
                    i_center, i_left, i_right = imr(_dir + center), imr(_dir + left), imr(_dir + right)
                    # flips of the images
                    i_center_f, i_left_f, i_right_f = np.fliplr(i_center), np.fliplr(i_left), np.fliplr(i_right)
                    # steering for center, left and right images
                    m_center, m_left, m_right = steer, steer + corr, steer - corr
                    # steering for flips
                    m_center_f, m_left_f, m_right_f = -steer, -(steer + corr), -(steer - corr)
                    # extend images
                    images.extend((i_center, i_left, i_right, i_center_f, i_left_f, i_right_f))
                    measurements.extend((m_center, m_left, m_right, m_center_f, m_left_f, m_right_f))
            features = np.array(images)
            labels = np.array(measurements)
            is_debugging = False
            if is_debugging:
                import cv2
                for feature in features:
                    cv2.imshow("center: ", feature)
                    cv2.waitKey()
            yield shuffle(features, labels)


def histogram_data(n_samples):
    x = n_samples
    plt.hist(x, bins=int(len(n_samples) / 10))
    plt.ylabel('Probability')
    plt.show()


def show_history(history):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def implement_model(shape):
    """
    it has 5 conv layers, 3 Dense layers and 1 output layer.
    filter depth, kernel and strides is taken from NVIDEA architecture specification.
    image is normalized and cropped before applying network on it.
    :param shape: shape of image
    :return: keras dnn model
    """

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=shape))  # normalize
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))  # cropping
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))  # layer 1
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))  # layer 2
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))  # layer 3
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # layer 4
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # layer 5
    model.add(Flatten())
    model.add(Dense(units=100))  # layer 6
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=50))  # layer 7
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=10))  # layer 8
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1))  # layer 9
    model.compile(optimizer='adam', loss='mse')
    return model


def load_disk_data(filename):
    cwd = os.getcwd()

    with open(cwd + filename, mode='rb') as f:
        data = pickle.load(f)

    features, labels = data['features'], data['labels']
    features, labels = np.array(features), np.array(labels)

    assert (len(features) == len(labels))
    shape = features.shape[1:]
    features, labels = shuffle(features, labels)

    print("shape: {}".format(shape))
    return features, labels, shape


def load_data(file_name, image_folder, is_debugging=True):
    cwd = os.getcwd()
    print("cwd: {}".format(cwd))
    samples = []
    with open(cwd + file_name) as filename:
        # read the log file
        reader = csv.reader(filename)
        # form an array of lines
        for line in reader:
            samples.append(line)
    # shuffle samples
    samples = shuffle(samples)
    images, measurements, steering = [], [], []
    # 0 --> center
    # 1 --> left
    # 2 --> right
    # save steering values
    for line in samples:
        steering.append(float(line[3]))
    # histogram of steering values
    if is_debugging:
        histogram_data(steering)
    for index, line in enumerate(samples):
        if index is 0: continue
        # append center camera images
        images, measurements = append_features_labels(image_folder,
                                                      line,
                                                      measurements,
                                                      images)

    features, labels = np.array(images), np.array(measurements)
    assert (len(features) == len(labels))

    shape = features.shape[1:]
    n_train = int(np.ceil(len(features) * 0.8))
    x_train, y_train = features[:n_train], labels[:n_train]
    x_validation, y_validation = features[n_train:], labels[n_train:]
    # print("Saving....")
    # save_data('train.p', x_train, y_train)
    # save_data('validation.p', x_validation, y_validation)
    # print("Saved")
    return features, labels, shape


def append_features_labels(_dir, line, measurements, images):
    center, left, right = line[0].split('/')[-1], line[1].split('/')[-1], line[2].split('/')[-1]
    steer, corr = float(line[3]), 0.2
    # images and flips
    if steer < 0:
        i_center, i_left, i_right = plt.imread(_dir + center), plt.imread(_dir + left), plt.imread(_dir + right)
        i_center_f, i_left_f, i_right_f = np.fliplr(i_center), np.fliplr(i_left), np.fliplr(i_right)
        is_debugging = False
        if is_debugging:
            import cv2
            print("shape: {}".format(i_center.shape))
            cv2.imshow("center: ", i_center[35:-12, :])
            cv2.imshow("left: ", i_left[35:-12, :])
            cv2.imshow("right: ", i_right[35:-12, :])
            cv2.imshow("center f: ", i_center_f[35:-12, :])
            cv2.imshow("left f: ", i_left_f[35:-12, :])
            cv2.imshow("right f: ", i_right_f[35:-12, :])
            cv2.waitKey()
        # measurements and flips
        m_center, m_left, m_right = steer, steer + corr, steer - corr
        m_center_f, m_left_f, m_right_f = -steer, -steer + corr, -steer - corr
        # extend the arrays
        images.extend((i_center, i_left, i_right))
        # i_center_f, i_left_f, i_right_f))
        measurements.extend((m_center, m_left, m_right))
        # m_center_f, m_left_f, m_right_f))

    return images, measurements


def save_data(filename, features, labels):
    assert (len(features) == len(labels))
    data = {
        'features': features,
        'labels': labels
    }
    pickle.dump(data, open(filename, "wb"))
    print("data saved to disk")
