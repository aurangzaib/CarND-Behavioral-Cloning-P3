import csv
import os
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Lambda, Dropout, Dense, Flatten, Cropping2D
from keras.models import Sequential
from matplotlib.image import imread as imr
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


def get_paths_and_steering(batch_sample):
    # center, left and right paths and steering
    center = batch_sample[0].split('/')[-1]
    left = batch_sample[1].split('/')[-1]
    right = batch_sample[2].split('/')[-1]
    steer = float(batch_sample[3])
    # steering correction
    corr = 0.2

    return center, left, right, steer, corr


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
                # center, left & right paths and steering
                center, left, right, steer, corr = get_paths_and_steering(batch_sample)
                # check for image file existence for given path in csv
                if exists(_dir + center) and exists(_dir + left) and exists(_dir + right):
                    # center, left and right images
                    im_center, im_left, im_right = imr(_dir + center), imr(_dir + left), imr(_dir + right)
                    # flips of the images
                    im_center_f, im_left_f, im_right_f = np.fliplr(im_center), np.fliplr(im_left), np.fliplr(im_right)
                    # steering for center, left and right images
                    m_center, m_left, m_right = steer, steer + corr, steer - corr
                    # steering for flips
                    m_center_f, m_left_f, m_right_f = -steer, -(steer + corr), -(steer - corr)
                    # extend images
                    images.extend((im_center, im_left, im_right, im_center_f, im_left_f, im_right_f))
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
    # normalize
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=shape))
    # cropping to reduce sky and other unnecessary features
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # layer 1
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    # layer 2
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    # layer 3
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    # layer 4
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    # layer 5
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    # layer 6
    model.add(Dense(units=100))
    model.add(Dropout(rate=0.5))
    # layer 7
    model.add(Dense(units=50))
    model.add(Dropout(rate=0.5))
    # layer 8
    model.add(Dense(units=10))
    model.add(Dropout(rate=0.5))
    # layer 9
    model.add(Dense(units=1))
    # adam optimizer and mse for error
    model.compile(optimizer='adam', loss='mse')

    return model
