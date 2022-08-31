import cv2


def get_contours(frame, mode, method):
    # print(cv2.findContours(frame, mode, method))
    x, contours, hierarchy = cv2.findContours(frame, mode, method)
    return contours


def get_biggest_contour(contours):
    max_cnt = contours[0]
    for cnt in contours:
        # print(cnt)
        if cv2.contourArea(cnt) > cv2.contourArea(max_cnt):
            max_cnt = cnt
    return max_cnt

def get_contour_area(contour):
    return cv2.contourArea(contour)

def compare_contours(contour_to_compare, saved_contours, max_diff):
    for contour in saved_contours:
        if cv2.matchShapes(contour_to_compare, contour, cv2.CONTOURS_MATCH_I2, 0) < max_diff:
            return True
    return False
    