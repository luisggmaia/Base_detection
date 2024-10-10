import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import clamp, sigmoid, isigmoid

class YOLO:
    """
    A YOLO detector class for targets detection
    """

    def __init__(self, model_name):
        """
        Constructs the YOLO detector based on the CNN.

        :param model_name: name of the model file
        :type model_name: str
        :param CNN_input_shape: input shape of the CNN
        :type CNN_input_shape: tuple(int, int, int)
        """

        self.model_name = model_name

        self.CNN_input_shape = (192, 256, 3)
        self.num_grid = 16 # 16 x 16 grid

        self.CNN_output_shape = (self.num_grid, self.num_grid, 5)

        self.anchor_box = (1, 1)

        try:
            self.CNN_model = models.load_model(self.model_name + '.hdf5', compile = False)
            self.CNN_model.compile(optimizer = 'adam', loss = self.yolo_loss, metrics = ['accuracy'])
        except OSError:
            print("Error while loading the model. Building a new one...")
            self.build_model( )
            self.CNN_model.summary( )

    def build_model(self):
        """
        Method to build the CNN model based on an specific architecture with 7 layers
        """

        self.CNN_model = models.Sequential( )

        self.CNN_model.add(layers.Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'conv_1', use_bias = False, input_shape = self.CNN_input_shape))
        self.CNN_model.add(layers.BatchNormalization(name = 'norm_1'))
        self.CNN_model.add(layers.LeakyReLU(alpha = 0.1, name = 'leaky_relu_1'))

        self.CNN_model.add(layers.Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'conv_2', use_bias = False))
        self.CNN_model.add(layers.BatchNormalization(name = 'norm_2'))
        self.CNN_model.add(layers.LeakyReLU(alpha = 0.1, name = 'leaky_relu_2'))

        self.CNN_model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'conv_3', use_bias = False))
        self.CNN_model.add(layers.BatchNormalization(name = 'norm_3'))
        self.CNN_model.add(layers.LeakyReLU(alpha = 0.1, name = 'leaky_relu_3'))

        self.CNN_model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'conv_4', use_bias = False))
        self.CNN_model.add(layers.BatchNormalization(name = 'norm_4'))
        self.CNN_model.add(layers.LeakyReLU(alpha = 0.1, name = 'leaky_relu_4'))
        self.CNN_model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', name = 'max_pool_4'))

        self.CNN_model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'conv_5', use_bias = False))
        self.CNN_model.add(layers.BatchNormalization(name = 'norm_5'))
        self.CNN_model.add(layers.LeakyReLU(alpha = 0.1, name = 'leaky_relu_5'))

        self.CNN_model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'conv_6', use_bias = False))
        self.CNN_model.add(layers.BatchNormalization(name = 'norm_6'))
        self.CNN_model.add(layers.LeakyReLU(alpha = 0.1, name = 'leaky_relu_6'))
        self.CNN_model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', name = 'max_pool_6'))

        self.CNN_model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'conv_7', use_bias = False))
        self.CNN_model.add(layers.BatchNormalization(name = 'norm_7'))
        self.CNN_model.add(layers.LeakyReLU(alpha = 0.1, name = 'leaky_relu_7'))
        self.CNN_model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', name = 'max_pool_7'))

        self.CNN_model.add(layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'conv_8', use_bias = False))
        self.CNN_model.add(layers.BatchNormalization(name = 'norm_8'))
        self.CNN_model.add(layers.LeakyReLU(alpha = 0.1, name = 'leaky_relu_8'))
        self.CNN_model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same', name = 'max_pool_8'))

        self.CNN_model.add(layers.Conv2D(filters = 5, kernel_size = (1, 1), strides = (1, 1), padding = 'same', name = 'conv_9', use_bias = True))

    def yolo_loss(self, y_corr, y_pred):
        """
        An simple YOLO loss function

        :param y_corr: correct return of the CNN
        :type y_corr: tf.Tensor
        :param y_pred: predicted return of the CNN
        :type y_pred: tf.Tensor
        :return: loss of the CNN
        :rtype: tf.Tensor
        """

        lambda_coord = 5
        lambda_wh = 5
        lambda_obj = 1
        lambda_noobj = 0.5
        
        q = 1/self.num_grid
        
        p_pred = tf.sigmoid(y_pred[:, :, :, 0])
        p_corr = tf.sigmoid(y_corr[:, :, :, 0])

        c = q*tf.tile(tf.expand_dims(tf.range(16), axis=0), [16, 1])
        
        b_x_pred = c + tf.sigmoid(y_pred[:, :, :, 1])
        b_y_pred = c + tf.sigmoid(y_pred[:, :, :, 2])
        b_w_pred = self.anchor_box[0]*tf.exp(y_pred[:, :, :, 3])
        b_h_pred = self.anchor_box[1]*tf.exp(y_pred[:, :, :, 4])
        
        b_x_corr = c + tf.sigmoid(y_corr[:, :, :, 1])
        b_y_corr = c + tf.sigmoid(y_corr[:, :, :, 2])
        b_w_corr = self.anchor_box[0]*tf.exp(y_corr[:, :, :, 3])
        b_h_corr = self.anchor_box[1]*tf.exp(y_corr[:, :, :, 4])
        
        coord_loss = lambda_coord*(tf.square(b_x_pred - b_x_corr) + tf.square(b_y_pred - b_y_corr)) + lambda_wh*(tf.square(tf.sqrt(b_w_pred) - tf.sqrt(b_w_corr)) + tf.square(tf.sqrt(b_h_pred) - tf.sqrt(b_h_corr)))
        
        conf_loss = lambda_obj*tf.square(p_pred - p_corr) + lambda_noobj*(1 - p_corr)*tf.square(p_pred)
        
        total_loss = tf.reduce_sum(coord_loss + conf_loss)

        return total_loss

    def preprocess_image(self, image, training = False):
        """
        Preprocess the image to be used in the CNN

        :param image: image to be preprocessed
        :type image: np.array
        :return: preprocessed image
        :rtype: np.array
        """

        image = cv2.resize(image, self.CNN_input_shape[:2], interpolation = cv2.INTER_AREA)
        image = np.array(image)
        image = image/255.0
        if training:
            image = np.reshape(image, self.CNN_input_shape)
        else:
            image = np.reshape(image, (1, *self.CNN_input_shape))

        return image

    def to_feature(self, boxes):
        """
        An auxiliar function to convert the num_targets bounding boxes of an image to the CNN expected output format

        :param boxes: list of bounding boxes
        :type boxes: list(tuple(int, int, int, int))
        :return: CNN expected output
        :rtype: np.array
        """

        corr_output = np.zeros(self.CNN_output_shape)

        q = 1/self.num_grid

        for box in boxes:

            b_x, b_y, b_w, b_h = box

            t_w = np.log(b_w/self.anchor_box[0])
            t_h = np.log(b_h/self.anchor_box[1])

            for i in range(self.num_grid):
                for j in range(self.num_grid):

                    t_x = isigmoid(b_x - j*q)
                    t_y = isigmoid(b_y - i*q)

                    clamped_x = clamp(b_x, j*q, (j + 1)*q)
                    clamped_y = clamp(b_y, i*q, (i + 1)*q)

                    a = min(clamped_x - b_x + b_w, b_x - clamped_x + b_w)
                    b = min(clamped_y - b_y + b_h, b_y - clamped_y + b_h)

                    if a > 0 and b > 0:
                        p = a*b/(2*b_w*b_h - a*b)
                    else:
                        p = 0.0
                    
                    t_o = isigmoid(p)

                    if corr_output[i, j, 0] < p:
                        corr_output[i, j, :] = [t_o, t_x, t_y, t_w, t_h]

        return corr_output

    def train_model(self, images, labels, epochs = 50, batch_size = 32):
        """
        Train the CNN model

        :param images: images to be used in the training
        :type images: np.array
        :param labels: labels of the images
        :type labels: np.array
        :param epochs: number of epochs
        :type epochs: int
        :param batch_size: size of the batch
        :type batch_size: int
        """

        self.CNN_model.compile(optimizer = 'adam', loss = self.yolo_loss, metrics = ['accuracy'])

        history = self.CNN_model.fit(images, labels, epochs = epochs, batch_size = batch_size)

        self.CNN_model.save(self.model_name + '.hdf5')

        plt.plot(history.history['loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Cost Function Convergence')
        plt.grid( )
        plt.show( )

    def process_output(self, CNN_output, min_confidence = 0.5):
        """
        Process the CNN output of an image to return the bounding boxes of the targets

        :param CNN_output: output of the CNN
        :type CNN_output: np.array
        :return: list of num_targets bounding boxes
        :rtype: list(tuple(int, int, int, int))
        """

        CNN_output = np.reshape(CNN_output, self.CNN_output_shape)

        boxes = [ ]

        q = 1/self.num_grid

        for i in range(self.num_grid):
            for j in range(self.num_grid):

                p = sigmoid(CNN_output[i, j, 0])

                if p >= min_confidence:

                    b_x = j*q + sigmoid(CNN_output[i, j, 1])
                    b_y = i*q + sigmoid(CNN_output[i, j, 2])

                    b_w = self.anchor_box[0]*np.exp(CNN_output[i, j, 3])
                    b_h = self.anchor_box[1]*np.exp(CNN_output[i, j, 4])

                    boxes.append((b_x, b_y, b_w, b_h))

        return boxes

    def detect(self, image):
        """
        Detects the targets in the image and returns the corresponding bounding boxes objects as a list

        :param image: image to be detected
        :type image: np.array
        :return: list of bounding boxes
        :rtype: list(tuple(int, int, int, int))
        """

        image = self.preprocess_image(image)
        CNN_output = self.CNN_model.predict(image)
        boxes = self.process_output(CNN_output)

        return boxes

