import csv
import os
import pickle

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread as imr
from sklearn.utils import shuffle


def visualize_histogram():
    """
    visualize the steering angles distribution using histogram
    :return:
    """
    cwd = os.getcwd()
    folder = '/data'
    csv_file = cwd + folder + '/driving_log.csv'
    # read the measurement from csv file
    samples = []
    # save samples
    corr = 0.2
    with open(csv_file) as file:
        for line in csv.reader(file):
            steering = float(line[3])
            if steering != 0:
                # include steering angles for left and right cameras
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
    fig.savefig('documentation/steering-distribution-center-left-right-flipped-updated.png')


def visualize_features():
    """
    visualize the steering angles distribution using histogram
    :return:
    """
    cwd = os.getcwd()
    folder = '/data'
    csv_file = cwd + folder + '/driving_log.csv'
    with open(csv_file) as file:
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
            im = imr("data/" + m)
            imf = np.fliplr(im)

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            plt.xticks([]), plt.yticks([])

            ax.set_title('Steering: {:.3f}'.format(s)), plt.imshow(im)
            fig.savefig("documentation/center-{}.png".format(count))

            ax.set_title('Steering: {:.3f}'.format(-s)), plt.imshow(imf)
            fig.savefig("documentation/center-flip-{}.png".format(count))
            count += 1

        count = 0
        for m, s in zip(left, str_left):
            im = imr("data/" + m)
            imf = np.fliplr(im)

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            plt.xticks([]), plt.yticks([])

            ax.set_title('Steering: {:.3f}'.format(s)), plt.imshow(im)
            fig.savefig("documentation/left-{}.png".format(count))

            ax.set_title('Steering: {:.3f}'.format(-s)), plt.imshow(imf)
            fig.savefig("documentation/left-flip-{}.png".format(count))
            count += 1

        count = 0
        for m, s in zip(right, str_right):
            im = imr("data/" + m)
            imf = np.fliplr(im)

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            plt.xticks([]), plt.yticks([])

            ax.set_title('Steering: {:.3f}'.format(s)), plt.imshow(im)
            fig.savefig("documentation/right-{}.png".format(count))

            ax.set_title('Steering: {:.3f}'.format(-s)), plt.imshow(imf)
            fig.savefig("documentation/right-flip-{}.png".format(count))
            count += 1


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
    for index, line in enumerate(samples):
        if index is 0:
            continue
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
    # center, left and right images
    im_center, im_left, im_right = imr(_dir + center), imr(_dir + left), imr(_dir + right)
    # flipped images
    im_center_f, im_left_f, im_right_f = np.fliplr(im_center), np.fliplr(im_left), np.fliplr(im_right)
    is_debugging = False
    if is_debugging:
        import cv2
        print("shape: {}".format(im_center.shape))
        cv2.imshow("center: ", im_center[35:-12, :])
        cv2.imshow("left: ", im_left[35:-12, :])
        cv2.imshow("right: ", im_right[35:-12, :])
        cv2.imshow("center f: ", im_center_f[35:-12, :])
        cv2.imshow("left f: ", im_left_f[35:-12, :])
        cv2.imshow("right f: ", im_right_f[35:-12, :])
        cv2.waitKey()
    # measurements and flips
    m_center, m_left, m_right = steer, steer + corr, steer - corr
    m_center_f, m_left_f, m_right_f = -steer, -steer + corr, -steer - corr
    # extend the arrays
    images.extend((im_center, im_left, im_right))
    # im_center_f, im_left_f, im_right_f))
    measurements.extend((m_center, m_left, m_right))
    # m_center_f, m_left_f, m_right_f))

    return images, measurements


def get_region_of_interest():
    top, bottom = 230, 135
    img = cv.imread("buffer/right-flip-8.png")
    height, width = img.shape[0], img.shape[1]
    res = img[int(top):int(top + bottom), int(0):int(width)]
    cv.imwrite("buffer/ROI-3.png", res)


def save_data(filename, features, labels):
    assert (len(features) == len(labels))
    data = {
        'features': features,
        'labels': labels
    }
    pickle.dump(data, open(filename, "wb"))
    print("data saved to disk")


get_region_of_interest()
