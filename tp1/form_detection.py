import cv2

from contour import get_contours, get_biggest_contour, compare_contours
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
    create_trackbar(trackbar_name, window_name, slider_max)
    create_trackbar(trackbar_name2, window_name, slider_max2)
    # saved_hu_moments = load_hu_moments(file_name="hu_moments.txt")
    saved_contours = []

    while True:
        ret, frame = cap.read()
        gray_frame = apply_color_convertion(frame=frame, color=cv2.COLOR_RGB2GRAY)
        trackbar_val = get_trackbar_value(trackbar_name=trackbar_name, window_name=window_name)
        trackbar_val2 = get_trackbar_value(trackbar_name=trackbar_name2, window_name=window_name)
        adapt_frame = adaptive_threshold(frame=gray_frame, slider_max=slider_max,
                                         adaptative=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         binary=cv2.THRESH_BINARY,
                                         trackbar_value=trackbar_val)
        frame_denoised = denoise(frame=adapt_frame, method=cv2.MORPH_ELLIPSE, radius=trackbar_val2)
        contours = get_contours(frame=frame_denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            biggest_contour = get_biggest_contour(contours=contours)
            # hu_moments = get_hu_moments(contour=biggest_contour)
            if compare_contours(contour_to_compare=biggest_contour, saved_contours=saved_contours, max_diff=1):
                draw_contours(frame=frame, contours=biggest_contour, color=color_white, thickness=20)
            draw_contours(frame=frame, contours=biggest_contour, color=color_white, thickness=3)

        cv2.imshow('Window', frame)
        if cv2.waitKey(1) & 0xFF == ord('k'):
            if biggest_contour is not None:
                # save_moment(hu_moments=hu_moments, file_name="hu_moments.txt")
                saved_contours.append(biggest_contour)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


main()