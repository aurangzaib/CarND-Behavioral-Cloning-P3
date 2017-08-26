import os

cwd = os.getcwd()


class Configuration():
    def __init__(self):
        # flags
        self.allow_data_flips = False
        self.is_debug_enabled = False
        self.use_pre_trained = True

        # input shape
        self.shape = (160, 320, 3)

        # folders of the training data
        self.folder1 = "../dataset-1"
        self.folder2 = "../dataset-2"
        self.folder3 = "../dataset-3"
        self.folder4 = "../dataset-4"

        # selected folder
        self.selected_folder = self.folder4

        # csv file
        self.selected_csv_file = self.selected_folder + '/driving_log.csv'

        # training images
        self.selected_img_file = self.selected_folder + '/IMG/'

        # documentation folders
        self.buffer_folder = "../buffer/"
        self.doc_folder = "../documentation/"

        # cropping image to feed classifier
        self.roi = (70, 25), (0, 0)

        # kernel for top conv layers
        self.kernel5 = (5, 5)

        # kernel for lower conv layers
        self.kernel3 = (3, 3)

        # activation function for cnn
        self.activation = "relu"

        # strides
        self.strides = (2, 2)

        # dropout keep rate
        self.rate = 0.5

        # optimizer for back propagation
        self.optimizer = "adam"
        self.loss = "mse"

        # model name
        self.model = '../model.h5'
