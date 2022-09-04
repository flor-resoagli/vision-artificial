import cv2 as cv
import csv
from math import sqrt, copysign, log10

from functions import denoise,draw_contour_train, return_label_name, return_label_number
final_data = []


def generate_hu(where_to_write, min, max):
    for i in range(min, max + 1):
        data = []
        image_path = './imgs/' + str(i) + '.png.png'
        image = cv.imread(image_path)
        image_colour = cv.imread(image_path)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        _, binary_img = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)
        denoised_img = denoise(binary_img, cv.MORPH_ELLIPSE, 20)
        moments = cv.moments(draw_contour_train(denoised_img, image_colour))
        huMoments = cv.HuMoments(moments)
        data.append(str(i))
        for j in range(0, 7):
            huMoments[j] = -1 * copysign(1.0, huMoments[j]) * log10(abs(huMoments[j]))
            data.append(huMoments[j][0])
        final_data.append(data)
        label = return_label_name(return_label_number(str(i)))
        cv.imshow(label, image_colour)
        cv.waitKey(0)

    with open(where_to_write, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(final_data)


generate_hu('./descriptores.csv', 2, 36)
