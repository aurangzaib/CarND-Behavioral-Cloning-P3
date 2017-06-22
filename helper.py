def load_data(file_name):
    from sklearn.utils import shuffle
    import numpy as np
    import cv2 as cv
    import csv
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    cwd = os.getcwd()
    print("cwd: {}".format(cwd))
    lines = []
    with open(cwd + file_name) as filename:
        # read the log file
        reader = csv.reader(filename)
        # form an array of lines
        for line in reader:
            lines.append(line)

    measurements, images = [], []

    for line in lines[:30000]:
        source_path = line[0]  # center image
        filename = source_path.split('/')[-1]
        image_path = cwd + '/data/IMG/' + filename
        if filename == 'center':
            continue
        # image --> features
        image = cv.imread(image_path)
        images.append(image)
        # steering --> labels
        measurement = float(line[3])
        measurements.append(measurement)

    features, labels = np.array(images), np.array(measurements)
    shape = features.shape[1:]

    print("shape: {}".format(shape))
    features, labels = shuffle(features, labels)
    return features, labels, shape
