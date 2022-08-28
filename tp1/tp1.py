from email.policy import default
import cv2 as cv


def load_shapes():

    # SQUARE
    gray_square = cv.cvtColor(cv.imread('shapes/square.jpeg'), cv.COLOR_RGB2GRAY)
    ret, square_threshold = cv.threshold(gray_square, 0, 255, 0)
    _, square_contours, _ = cv.findContours(square_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # TRIANGLE
    gray_triangle = cv.cvtColor(cv.imread('shapes/triangle.jpeg'), cv.COLOR_RGB2GRAY)
    ret, triangle_threshold = cv.threshold(gray_triangle, 0, 255, 0)
    _, triangle_contours, _ = cv.findContours(triangle_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # CIRCLE
    gray_circle = cv.cvtColor(cv.imread('shapes/circle.jpeg'), cv.COLOR_RGB2GRAY)
    ret, circle_threshold = cv.threshold(gray_circle, 0, 255, 0)
    _, circle_contours, _ = cv.findContours(circle_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    shapes = [square_contours, triangle_contours, circle_contours]

    return shapes

def get_binary_image(image, value):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret, threshold = cv.threshold(gray_image, value, 255, cv.THRESH_BINARY_INV)
    return threshold
    
def get_denoised_image(binary):
    structuring_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    morph_open = cv.morphologyEx(binary, cv.MORPH_OPEN, structuring_element)
    morph_close = cv.morphologyEx(binary, cv.MORPH_CLOSE, structuring_element)
    return morph_close

def get_biggest_contour(contours):
    max_cnt = contours[1]
    for cnt in contours:
        # print(cnt)
        if cv.contourArea(cnt) > cv.contourArea(max_cnt):
            max_cnt = cnt
    return max_cnt

def main():

    # SQUARE
    gray_square = cv.cvtColor(cv.imread('shapes/square.jpeg'), cv.COLOR_RGB2GRAY)
    ret, square_threshold = cv.threshold(gray_square, 0, 255, 0)
    _, square_contours, _ = cv.findContours(square_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # TRIANGLE
    gray_triangle = cv.cvtColor(cv.imread('shapes/triangle.jpeg'), cv.COLOR_RGB2GRAY)
    ret, triangle_threshold = cv.threshold(gray_triangle, 0, 255, 0)
    _, triangle_contours, _ = cv.findContours(triangle_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # CIRCLE
    gray_circle = cv.cvtColor(cv.imread('shapes/circle.jpeg'), cv.COLOR_RGB2GRAY)
    ret, circle_threshold = cv.threshold(gray_circle, 0, 255, 0)
    _, circle_contours, _ = cv.findContours(circle_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    webcam = cv.VideoCapture(0)
    cv.namedWindow('Binary')
    # cv.createTrackbar('Binary_Trackbar', 'Binary', 0, 100, (lambda a: None))

    while True:

        #original image
        ret, original_image = webcam.read()
        # cv.imshow('Original', original_image)

        #binary image
        # binary_value = cv.getTrackbarPos('Binary_Trackbar', 'Binary')
        binary_image = get_binary_image(original_image, 100)
        cv.imshow('Binary', binary_image)

        #denoised image
        denoised_image = get_denoised_image(binary_image)
        cv.imshow('Denoised', denoised_image)

        #contours
        _, contours, _ = cv.findContours(denoised_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)


        #filtered contours
        if len(contours) > 0:
            contours = [c for c in contours if cv.contourArea(c) > 5000]
            contours.pop(0)
            # cv.drawContours(image=original_image, contours=[get_biggest_contour(contours)], contourIdx=-1, color=(0, 0, 255), thickness=3)

            

        # #compare with references
        for c in contours:
            
            match_square = cv.matchShapes(c, square_contours[0], cv.CONTOURS_MATCH_I2, 0)
            match_triangle = cv.matchShapes(c, triangle_contours[0], cv.CONTOURS_MATCH_I2, 0)
            match_circle = cv.matchShapes(c, circle_contours[0], cv.CONTOURS_MATCH_I2, 0)

            min_match = min(match_square, match_triangle, match_circle)

            if min_match < 0.07:
                cv.drawContours(image=original_image, contours=[c], contourIdx=-1, color=(0, 255, 0), thickness=3)
                x, y, w, h = cv.boundingRect(c)
                text = 'Shape'

                cv.putText(original_image, text, (x, y), cv.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)
            

        cv.imshow('Original', original_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()


main()