import csv
import os
from os.path import exists

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from matplotlib.image import imread as imr
from numpy import fliplr as flp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from classifier import Classifier
from configuration import Configuration

conf = Configuration().__dict__
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Helper:
    @staticmethod
    def load_samples(quantity=-1):
        """
        return lines from csv file
        each line contains left, center, right images and steering info
        @:param csv_file: csv file path
        @:param quantity: how many lines to be returned from csv file
        """
        samples = []
        # save samples
        with open(conf["selected_csv_file"]) as file:
            for line in csv.reader(file):
                samples.append(line)
        # shuffle and return samples
        samples = shuffle(samples)
        samples = samples if quantity is -1 else samples[:quantity]

        print("total samples: {}".format(len(samples)))
        return train_test_split(samples, test_size=0.2)

    @staticmethod
    def get_paths_and_steering(batch_sample):
        # center, left and right paths and steering
        center = batch_sample[0].split('/')[-1]
        left = batch_sample[1].split('/')[-1]
        right = batch_sample[2].split('/')[-1]
        steer = float(batch_sample[3])
        # steering correction
        corr = 0.2

        return center, left, right, steer, corr

    @staticmethod
    def generator(samples, batch_size=32):
        """
        generator --> 1- infinite while loop 2- yield instead of return
        :param samples: lines from csv file
        :param batch_size: how many features and labels each time
        :return: features, labels
        """
        imdir = conf["selected_img_file"]
        num_samples = len(samples)

        # Loop forever so the generator never terminates
        while 1:
            for offset in range(0, num_samples, batch_size):

                batch_samples = samples[offset:offset + batch_size]
                images, measurements = [], []

                for batch_sample in batch_samples:

                    # center, left & right paths and steering
                    center, left, right, steer, corr = Helper.get_paths_and_steering(batch_sample)

                    # check for image file existence for given path in csv
                    if exists(imdir + center) and exists(imdir + left) and exists(imdir + right):

                        # center, left and right images
                        im_center, im_left, im_right = imr(imdir + center), imr(imdir + left), imr(imdir + right)
                        # steering for center, left and right images
                        m_center, m_left, m_right = steer, steer + corr, steer - corr

                        # extend images
                        if conf["allow_data_flips"] is True:
                            # flips of the images
                            im_center_f, im_left_f, im_right_f = flp(im_center), flp(im_left), flp(im_right)

                            # steering for flips
                            m_center_f, m_left_f, m_right_f = -steer, -(steer + corr), -(steer - corr)

                            # extend measurements and images with flips
                            images.extend((im_center_f, im_left_f, im_right_f))
                            measurements.extend((m_center_f, m_left_f, m_right_f))

                        # extend measurements and images with originals
                        images.extend((im_center, im_left, im_right))
                        measurements.extend((m_center, m_left, m_right))

                features = np.array(images)
                labels = np.array(measurements)

                if conf["is_debug_enabled"]:
                    for feature in features:
                        cv.imshow("center: ", feature)
                        cv.waitKey()
                yield shuffle(features, labels)

    @staticmethod
    def show_history(history):
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Metrics')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()

    @staticmethod
    def get_model_summary(model):
        for layer in model.layers:
            print(layer.get_weights())
        print("model summary: \n{}\n".format(model.summary()))
        print("model parameters: \n{}\n".format(model.count_params()))

    @staticmethod
    def save_model(model, history):
        # 1- architecture
        # 2- weights
        # 3- optimizer and loss
        # 4- used for transfer learning with load_model
        model.save(conf["model"])
        Helper.show_history(history.history)

    @staticmethod
    def get_model():
        model = load_model('../model.h5') if conf["use_pre_trained"] else Classifier.implement_classifier()
        return model
