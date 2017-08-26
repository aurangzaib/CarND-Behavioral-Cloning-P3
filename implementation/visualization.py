import csv

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread as imr
from sklearn.utils import shuffle

from configuration import Configuration

conf = Confuration().__dict__


class Visualization:
    @staticmethod
    def visualize_histogram():
        """
        visualize the steering angles distribution using histogram
        :return: None
        """
        # read the measurement from csv file
        samples = []
        # save samples
        corr = 0.2
        with open(conf["csv_file"]) as file:
            for line in csv.reader(file):
                steering = float(line[3])
                if steering != 0:
                    samples.extend((steering, steering + corr, steering - corr))
                    samples.extend((-steering, -(steering + corr), -(steering - corr)))

        unique_classes, n_samples = np.unique(samples,
                                              return_index=False,
                                              return_inverse=False,
                                              return_counts=True)

        mu, sigma = np.mean(samples), np.std(samples)
        print("mean: {}, std: {}".format(mu, sigma))

        width = 0.01  # 1 / len(unique_classes)
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.set_title('Samples Distribution')
        ax.set_xlabel('Steering Angle')
        ax.set_ylabel('Number of Samples')
        plt.bar(unique_classes, n_samples, width, color="blue")
        fig.savefig('{}steering-distribution-center-left-right-flipped.png'.format(conf["buffer_folder"]))

    @staticmethod
    def visualize_features():
        """
        visualize the steering angles distribution using histogram
        :return:
        """
        with open(conf["csv_file"]) as file:
            samples = []
            corr = 0.2

            # store all samples
            for line in csv.reader(file):
                samples.append(line)

            # shuffle samples and take only 10 of them randomly
            samples = (shuffle(samples))

            center, left, right = [], [], []
            str_center, str_left, str_right = [], [], []

            for sample in samples:
                steering = float(sample[3])
                if steering < -.1:
                    left.append(sample[0])
                    str_left.append(steering)
                elif steering > .1:
                    right.append(sample[0])
                    str_right.append(steering)
                else:
                    center.append(sample[0])
                    str_center.append(steering)

            center = center[:10]
            left = left[:10]
            right = right[:10]

            count = 0
            for m, s in zip(center, str_center):
                im = imr(conf["folder"] + m)
                imf = np.fliplr(im)

                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111)
                plt.xticks([]), plt.yticks([])

                ax.set_title('Steering: {:.3f}'.format(s)), plt.imshow(im)
                fig.savefig("{}center-{}.png".format(conf["doc_folder"], count))

                ax.set_title('Steering: {:.3f}'.format(-s)), plt.imshow(imf)
                fig.savefig("{}center-flip-{}.png".format(conf["doc_folder"], count))
                count += 1

            count = 0
            for m, s in zip(left, str_left):
                im = imr(conf["folder"] + m)
                imf = np.fliplr(im)

                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111)
                plt.xticks([]), plt.yticks([])

                ax.set_title('Steering: {:.3f}'.format(s)), plt.imshow(im)
                fig.savefig("{}left-{}.png".format(conf["doc_folder"], count))

                ax.set_title('Steering: {:.3f}'.format(-s)), plt.imshow(imf)
                fig.savefig("{}left-flip-{}.png".format(conf["doc_folder"], count))
                count += 1

            count = 0
            for m, s in zip(right, str_right):
                im = imr(conf["folder"] + m)
                imf = np.fliplr(im)

                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111)
                plt.xticks([]), plt.yticks([])

                ax.set_title('Steering: {:.3f}'.format(s)), plt.imshow(im)
                fig.savefig("{}right-{}.png".format(conf["doc_folder"], count))

                ax.set_title('Steering: {:.3f}'.format(-s)), plt.imshow(imf)
                fig.savefig("{}right-flip-{}.png".format(conf["doc_folder"], count))
                count += 1

    @staticmethod
    def visualize_roi():
        top, bottom = 230, 135
        img = cv.imread("{}right-flip-8.png".format(conf["buffer_folder"]))
        height, width = img.shape[0], img.shape[1]
        res = img[int(top):int(top + bottom), int(0):int(width)]
        cv.imwrite("{}ROI-3.png".format(conf["buffer_folder"]), res)

        # @staticmethod
        # def training_data_summary():


Visualization.visualize_histogram()
