import os
import cv2
import numpy as np
from YOLO import YOLO

def get_training_inputs(images, labels, num_images):
    """
    Format the inputs for the training of the CNN model.

    :param images: images to be used for training.
    :type images: list(np.array)
    :param labels: labels of the images.
    :type labels: list(np.array)
    :param num_images: number of images.
    :type num_images: int
    :return: formatted images and labels.
    :rtype: np.array, np.array
    """

    images_input = np.array([yolo.preprocess_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), training = True) for image in images], dtype = np.float32)
    labels_input = np.array([yolo.to_feature(boxes) for boxes in labels], dtype = np.float32)

    random_indices = np.arange(num_images)
    np.random.shuffle(random_indices)

    images_input = images_input[random_indices]
    labels_input = labels_input[random_indices]

    return images_input, labels_input

yolo_model_name = 'Model\\CNN'

yolo = YOLO(yolo_model_name)

labels_file_name = 'img_'
images_name = 'img_'

num_images = 1000
epochs = 10

boxes = [ ]
labels = [ ]

for n in range(num_images):

    with open(f"Labels\\{labels_file_name}{n}.txt", 'r') as set_file:

        for line in set_file:

            box = list(map(float, line.split( )))

            boxes.append(box[1:])
        
        labels.append(boxes)

images = [cv2.imread(f"dataset\\{images_name}{n}.png") for n in range(num_images)]

images, labels = get_training_inputs(images, labels, num_images)

yolo.train_model(images, labels, epochs)

