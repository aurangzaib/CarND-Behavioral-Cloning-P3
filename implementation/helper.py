import csv
import os
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
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
