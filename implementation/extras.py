import csv
import os
import pickle

import numpy as np
from matplotlib.image import imread as imr
from sklearn.utils import shuffle


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


def load_data(file_name, image_folder):
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

    print("Saving....")
    save_data('train.p', x_train, y_train)
    save_data('validation.p', x_validation, y_validation)
    print("Saved")

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
    # extend the arrays
    images.extend((im_center, im_left, im_right))
    # im_center_f, im_left_f, im_right_f))
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
