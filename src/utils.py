import cv2
from imutils import face_utils
import imutils
import dlib

from os import listdir

import math
import numpy as np
import pandas as pd
from scipy import optimize


# # Define text info
# font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale = 1
# fontColor = (255, 255, 255)
# lineType = 2

def predict_landmarks(image_path, landmark_predictor, face_detector):
    """
    Predicting the landmarks given a image filepath and a predictor object
    nput:
        - image_path     : image file path
        - predictor      : predictor object
    return:
        - image          : output image
        - shape          : list of landmark locations
    """

    # face_detector = dlib.get_frontal_face_detector()
    image = cv2.imread(image_path)
    output = image.copy()

    # image = imutils.resize(image, width = 250)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    rects = face_detector(gray, 1)
    if len(rects) == 0:
        rects = dlib.rectangles()
        h, w = image.shape[0], image.shape[1]
        rec = dlib.rectangle(0, 0, w, h)
        rects.append(rec)

    shape = 0
    # loop over the face detections
    for (i, rect) in enumerate(rects):

        shape = landmark_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1)
    # return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), shape
    return image, output, shape
    # return image, shape

# def detect_eye(image, shape):
#     # print(shape)
#     left_x = [i[0] for i in shape[:6]]
#     left_y = [i[1] for i in shape[:6]]
#     left_size = int((max(left_x) - min(left_x)) * 2.5)
#     left_top = int(np.mean(left_y) - left_size / 2)
#     left_left = int((max(left_x) + min(left_x) - left_size) / 2)
#     left_eye = image[left_top: left_top + left_size, \
#                      left_left: left_left + left_size]
#
#     right_x = [i[0] for i in shape[6:]]
#     right_y = [i[1] for i in shape[6:]]
#     right_size = int((max(right_x) - min(right_x)) * 2.5)
#     right_top = int(np.mean(right_y) - right_size / 2)
#     right_left = int((max(right_x) + min(right_x) - right_size) / 2)
#     right_eye = image[right_top: right_top + right_size, \
#                      right_left: right_left + right_size]
#
#     return left_eye, right_eye


# Function to resize image
def resize_to(desired_size, im_pth):
    im = cv2.imread(im_pth, 0)
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size // 2 - new_size[0]
    top, bottom = delta_h // 2, delta_h-(delta_h//2)
    left, right = delta_w // 2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    try:
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value = color)
        return new_im
    except:
        print('  => Error at:', im_pth)
        return

def get_random_string(length):
    letters = list(string.ascii_lowercase)
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
