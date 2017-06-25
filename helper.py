def load_disk_data(filename):
    from sklearn.utils import shuffle
    import numpy as np
    import pickle
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    from sklearn.utils import shuffle
    import numpy as np
    import csv
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    cwd = os.getcwd()
    print("cwd: {}".format(cwd))
    samples = []
    with open(cwd + file_name) as filename:
        # read the log file
        reader = csv.reader(filename)
        # form an array of lines
        for line in reader:
            samples.append(line)
    # samples = shuffle(samples)
    measurements, images = [], []
    # 0 --> center
    # 1 --> left
    # 2 --> right
    steering = []
    for line in samples:
        steering.append(float(line[3]))
    histogram_data(steering)
    for index, line in enumerate(samples):
        if index is 0:
            continue
        # append center camera images
        images, measurements = append_features_labels(cwd + image_folder, line, measurements, images)

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
    import matplotlib.pyplot as plt
    import numpy as np
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
    import pickle
    assert (len(features) == len(labels))
    data = {
        'features': features,
        'labels': labels
    }
    pickle.dump(data, open(filename, "wb"))
    print("data saved to disk")


def show_history(history):
    import matplotlib.pyplot as plt
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def generator(_dir, samples, batch_size=32):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.utils import shuffle
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                steer, corr = float(batch_sample[3]), 0.2
                center = batch_sample[0].split('/')[-1]
                left = batch_sample[1].split('/')[-1]
                right = batch_sample[2].split('/')[-1]

                i_center, i_left, i_right = plt.imread(_dir + center), plt.imread(_dir + left), plt.imread(_dir + right)
                i_center_f, i_left_f, i_right_f = np.fliplr(i_center), np.fliplr(i_left), np.fliplr(i_right)

                m_center, m_left, m_right = steer, steer + corr, steer - corr
                m_center_f, m_left_f, m_right_f = -steer, -steer - corr, -steer + corr

                images.extend((i_center, i_left, i_right,
                               i_center_f, i_left_f, i_right_f))
                measurements.extend((m_center, m_left, m_right,
                                     m_center_f, m_left_f, m_right_f))

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
    import matplotlib.pyplot as pl
    x = n_samples
    pl.hist(x, bins=int(len(n_samples) / 10))
    pl.ylabel('Probability')
    pl.show()


def load_samples(csv_file, quantity):
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    import csv
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    cwd = os.getcwd()
    samples = []
    with open(cwd + csv_file) as file:
        reader = csv.reader(file)
        for line in reader:
            samples.append(line)
    # 2 --> flips
    # 3 --> images per line
    samples = shuffle(samples)
    samples = samples[:quantity]
    return train_test_split(samples, test_size=0.2)
