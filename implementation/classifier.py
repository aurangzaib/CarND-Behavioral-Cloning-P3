from keras.layers import Conv2D, Lambda, Dropout, Dense, Flatten, Cropping2D
from keras.models import Sequential


class Classifier:
    @staticmethod
    def implement_classifier(shape):
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
