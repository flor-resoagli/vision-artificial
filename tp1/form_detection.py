import cv2

from contour import get_contours, get_biggest_contour, compare_contours, get_contour_area
from frame_editor import apply_color_convertion, adaptive_threshold, denoise, draw_contours
from trackbar import create_trackbar, get_trackbar_value


def main():

    window_name = 'Window'
    trackbar_name = 'Trackbar'
    trackbar_name2 = 'Trackbar2'
    slider_max = 151
    slider_max2 = 20
    cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(0)
    biggest_contour = None
    color_white = (255, 255, 255)
    color_blue = (200, 150, 100)
    create_trackbar(trackbar_name, window_name, slider_max)
    create_trackbar(trackbar_name2, window_name, slider_max2)

    saved_contours = []

    while True:
        ret, frame = cap.read()
        gray_frame = apply_color_convertion(frame=frame, color=cv2.COLOR_RGB2GRAY)
        trackbar_val = get_trackbar_value(trackbar_name=trackbar_name, window_name=window_name)
        trackbar_val2 = get_trackbar_value(trackbar_name=trackbar_name2, window_name=window_name)
        adapt_frame = adaptive_threshold(frame=gray_frame, slider_max=slider_max, adaptative=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, binary=cv2.THRESH_BINARY,trackbar_value=trackbar_val)
        frame_denoised = denoise(frame=adapt_frame, method=cv2.MORPH_ELLIPSE, radius=trackbar_val2)
        contours = get_contours(frame=frame_denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        ###
        rectangle = cv2.imread('../tp1/rectangle.jpg')
        img = cv2.cvtColor(rectangle, cv2.COLOR_BGR2GRAY)
        ret2, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        _,contour_rect, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt1=contour_rect
        ###

        if len(contours) > 0:
            biggest_contour = get_biggest_contour(contours=contours)
            if compare_contours(contour_to_compare=biggest_contour, saved_contours=saved_contours, max_diff=1):
                draw_contours(frame=frame_denoised, contours=biggest_contour, color=color_blue, thickness=20)
                draw_contours(frame=frame, contours=biggest_contour, color=color_blue, thickness=20)

            for c in contours:
                if get_contour_area(c)>5000:
                    draw_contours(frame=frame_denoised, contours=[c], color=color_blue, thickness=3)        
                    draw_contours(frame=frame, contours=[c], color=color_blue, thickness=3)  
                    if cv2.matchShapes(c, cnt1[1], cv2.CONTOURS_MATCH_I2, 0) < 0.4:
                        cv2.drawContours(frame, [c],-1, (0, 0, 255), 2)      
           
        cv2.imshow('Window', frame_denoised)
        cv2.imshow('Window2',frame )
        if cv2.waitKey(1) & 0xFF == ord('k'):
            if biggest_contour is not None:
                saved_contours.append(biggest_contour)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


main()
