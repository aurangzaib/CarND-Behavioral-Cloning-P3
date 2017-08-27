import csv
import glob

import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread as imr
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from configuration import Configuration

conf = Configuration().__dict__


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
        folders = [conf["folder1"], conf["folder2"], conf["folder3"], conf["folder4"]]
        for folder in folders:
            csv_file = folder + '/driving_log.csv'
            with open(csv_file) as file:
                for line in csv.reader(file):
                    steering = float(line[3])
                    if steering != 0:
                        samples.append(steering)
                        samples.extend((steering, steering + corr, steering - corr))
                        samples.extend((-steering, -(steering + corr), -(steering - corr)))
            print("samples: {}".format(len(samples)))

        mu, sigma = np.mean(samples), np.std(samples)
        print("mean: {}, std: {}".format(mu, sigma))

        samples = np.array(samples)
        samples = samples.reshape(-1, 1)

        samples = StandardScaler().fit_transform(samples)

        mu, sigma = np.mean(samples), np.std(samples)
        print("mean norm: {}, std norm: {}".format(mu, sigma))

        unique_classes, n_samples = np.unique(samples,
                                              return_index=False,
                                              return_inverse=False,
                                              return_counts=True)

        width = 0.01  # 1 / len(unique_classes)
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        ax.set_title('Samples Distribution')
        ax.set_xlabel('Steering Angle')
        ax.set_ylabel('Number of Samples')

        plt.bar(unique_classes, n_samples, width, color="blue")
        fig.savefig('{}steering-distribution-augmented-all-cameras-flips-0-removed-normalized.png'.format(
            conf["buffer_folder"]))

    # steering-distribution-center-left-right-flipped
    @staticmethod
    def visualize_features(folder="../dataset-1/"):
        """
        visualize the steering angles distribution using histogram
        :return:
        """
        with open("{}driving_log.csv".format(folder)) as file:
            samples = []
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
                im = imr(folder + m)
                imf = np.fliplr(im)

                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111)
                plt.xticks([]), plt.yticks([])

                ax.set_title('Steering: {:.3f}'.format(s)), plt.imshow(im)
                fig.savefig("{}center-{}.png".format(conf["buffer_folder"] + "track2/", count))

                ax.set_title('Steering: {:.3f}'.format(-s)), plt.imshow(imf)
                fig.savefig("{}center-flip-{}.png".format(conf["buffer_folder"] + "track2/", count))
                count += 1

            count = 0
            for m, s in zip(left, str_left):
                im = imr(folder + m)
                imf = np.fliplr(im)

                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111)
                plt.xticks([]), plt.yticks([])

                ax.set_title('Steering: {:.3f}'.format(s)), plt.imshow(im)
                fig.savefig("{}left-{}.png".format(conf["buffer_folder"] + "track2/", count))

                ax.set_title('Steering: {:.3f}'.format(-s)), plt.imshow(imf)
                fig.savefig("{}left-flip-{}.png".format(conf["buffer_folder"] + "track2/", count))
                count += 1

            count = 0
            for m, s in zip(right, str_right):
                im = imr(folder + m)
                imf = np.fliplr(im)

                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111)
                plt.xticks([]), plt.yticks([])

                ax.set_title('Steering: {:.3f}'.format(s)), plt.imshow(im)
                fig.savefig("{}right-{}.png".format(conf["buffer_folder"] + "track2/", count))

                ax.set_title('Steering: {:.3f}'.format(-s)), plt.imshow(imf)
                fig.savefig("{}right-flip-{}.png".format(conf["buffer_folder"] + "track2/", count))
                count += 1

    @staticmethod
    def visualize_roi():
        top, bottom = 230, 135
        left, right = 70, 70
        track = "track1"
        img = cv.imread("{}{}/right-flip-4.png".format(conf["buffer_folder"], track))
        height, width = img.shape[0], img.shape[1]
        res = img[int(top):int(top + bottom), int(left):int(width - right)]
        cv.imwrite("{}{}/ROI-4.png".format(conf["buffer_folder"], track), res)

    @staticmethod
    def reduce_images_height():
        top, bottom = 70, 70
        left, right = 70, 70
        imgs = glob.glob("{}data-exploration*".format(conf["doc_folder"]))
        for filename in imgs:
            im = imr(filename)
            im_roi = np.copy(im)
            height, width = im_roi.shape[0], im_roi.shape[1]
            res = im_roi[int(top):int(height - bottom), int(left):int(width - right)]
            mpimg.imsave(filename, res)


Visualization.visualize_histogram()
