import os
import cv2
from YOLO import YOLO
import matplotlib.pyplot as plt

RECT_THICKNESS = 3
POINT_THICKNESS = 3

def show_detection(image, boxes, color, show_center = True):
    """
    Shows an object detection.

    :param image: image used for object detection.
    :type image: OpenCV's image.
    :param detection: detection as a 5-dimensional tuple: (probability, x, y, width, height).
    :type detection: 5-dimensional tuple.
    :param color: color used to show the detection (RGB value).
    :type color: 3-dimensional tuple.
    :param show_center: if the center of the detection should be shown in the image.
    :type show_center: bool.
    """
    x = boxes[0]
    y = boxes[1]
    width = boxes[2]
    height = boxes[3]
    top_left = (int(x - width/2), int(y - height/2))
    bottom_right = (int(x + width/2), int(y + height/2))
    cv2.rectangle(image, top_left, bottom_right, color, RECT_THICKNESS)
    if show_center:
        cv2.circle(image, (int(x), int(y)), POINT_THICKNESS, color, -1)

yolo_model_name = 'Model\\CNN'

yolo = YOLO(yolo_model_name)

evaluate_images_name = 'img_'

num_evaluate_images = 20 # Put the number of images you want to evaluate
num_images_show = 20 # Put the number of images you want to show

evaluate_images = [cv2.imread(f"evaluation_dataset\\{evaluate_images_name}{n}.png") for n in range(num_evaluate_images)]

for n in range(num_images_show):

    image = evaluate_images[n]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = yolo.detect(image)

    for box in boxes:
        show_detection(image, box, (0, 255, 0))

    plt.figure( )
    plt.imshow(image)
    plt.show( )

