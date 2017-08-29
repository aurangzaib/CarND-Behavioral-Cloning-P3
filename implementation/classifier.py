from keras.layers import Conv2D, Lambda, Dropout, Dense, Flatten, Cropping2D
from keras.models import Sequential

from configuration import Configuration

conf = Configuration().__dict__


class Classifier:
    @staticmethod
    def implement_classifier():
        """
        it has 5 conv layers, 3 Dense layers and 1 output layer.
        filter depth, kernel and strides is taken from NVIDEA architecture specification.
        image is normalized and cropped before applying network on it.
        :return: keras dnn model
        """
        model = Sequential()
        shape = conf["shape"]
        # cropping to reduce sky and other unnecessary features
        model.add(Cropping2D(cropping=(conf["roi"])))
        # normalize
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=shape))
        # layer 1
        model.add(Conv2D(filters=24, kernel_size=conf["kernel5"],
                         strides=conf["strides"], activation=conf["activation"]))
        # layer 2
        model.add(Conv2D(filters=36, kernel_size=conf["kernel5"],
                         strides=conf["strides"], activation=conf["activation"]))
        # layer 3
        model.add(Conv2D(filters=48, kernel_size=conf["kernel5"],
                         strides=conf["strides"], activation=conf["activation"]))
        # layer 4
        model.add(Conv2D(filters=64, kernel_size=conf["kernel3"],
                         activation=conf["activation"]))
        # layer 5
        model.add(Conv2D(filters=64, kernel_size=conf["kernel3"],
                         activation=conf["activation"]))
        model.add(Flatten())
        # layer 6
        model.add(Dense(units=100))
        model.add(Dropout(rate=conf["rate"]))
        # layer 7
        model.add(Dense(units=50))
        model.add(Dropout(rate=conf["rate"]))
        # layer 8
        model.add(Dense(units=10))
        model.add(Dropout(rate=conf["rate"]))
        # layer 9
        model.add(Dense(units=1))
        # update weights with adam optimizer and mse for error
        model.compile(optimizer=conf["optimizer"], loss=conf["loss"])

        return model
