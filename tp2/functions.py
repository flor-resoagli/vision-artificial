import csv

import cv2 as cv


def empty(val):
    pass


def binary(val, frame):
    # blur_frame = cv.GaussianBlur(frame, (7, 7), 1)
    gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    _, binary_img = cv.threshold(gray_frame, val, 255, cv.THRESH_BINARY_INV)
    return binary_img


def filter_screen_contour(contours, value):
    new_contours = []
    for cnt in contours:
        if cv.contourArea(cnt) < value:
            new_contours.append(cnt)
    return new_contours


def get_trackbar_value(trackbar_name, window_name):
    return int(cv.getTrackbarPos(trackbar_name, window_name) / 2) * 2 + 3


def denoise(frame, method, radius):
    kernel = cv.getStructuringElement(method, (radius, radius))
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    return closing


def get_biggest_contour(contours):
    max_contour = contours[0]
    for contour in contours:
        if 8000 > cv.contourArea(contour) > cv.contourArea(max_contour):
            max_contour = contour
    return max_contour


def draw_contour_train(contours_from, contours_to):
    _, contours, hierarchy = cv.findContours(contours_from, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if cv.contourArea(contours[1]) > 8000 and cv.contourArea(contours[0]) > 8000:
        cnt =  contours[1]
    else:
        cnt = contours[2]
    
    cv.drawContours(contours_to, cnt, -1, (0, 255, 0), 7)
    return cnt


def draw_contours(contours_from, contours_to, value):
    contours, hierarchy = cv.findContours(contours_from, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    no_screen_contours = filter_screen_contour(contours, 10000)
    new_contours = []

    for cnt in no_screen_contours:
        area = cv.contourArea(cnt)
        if area > value:
            new_contours.append(cnt)
            # cv.drawContours(contours_to, cnt, -1, (255, 0, 255), 7)

    return new_contours


def return_label_number(photo_number):
    with open('./supervisión.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == photo_number:
                return row[1]


def return_label_name(label_number):
    with open('./etiquetas.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == label_number:
                return str(row[1])
