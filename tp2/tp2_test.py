
import cv2
from joblib import load
from math import copysign, log10

from contour import get_contours, get_biggest_contour, compare_contours, get_contour_area
from frame_editor import apply_color_convertion, adaptive_threshold, denoise, draw_contours
from trackbar import create_trackbar, get_trackbar_value

# carga el modelo
clasificador = load('filename.joblib') 

# etiquetaPredicha = clasificador.predict(invariantesDeHu)

def get_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    for i in range(len(hu_moments)):
        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
    return hu_moments

def main():

    window_name = 'Window'
    cv2.namedWindow(window_name)
    create_trackbar("BINARY", window_name, 151)
    create_trackbar("DENOISE", window_name, 20)
    create_trackbar("MATCH", window_name, 10)

    cap = cv2.VideoCapture(0)
    biggest_contour = None

    while True:

        ret, frame = cap.read()

        binary_val = get_trackbar_value(trackbar_name="BINARY", window_name=window_name)
        denoise_val = get_trackbar_value(trackbar_name="DENOISE", window_name=window_name)
        match_val = get_trackbar_value(trackbar_name="MATCH", window_name=window_name)

        gray_frame = apply_color_convertion(frame=frame, color=cv2.COLOR_RGB2GRAY)

        ret2, adapt_frame = cv2.threshold(gray_frame, binary_val, 255, cv2.THRESH_BINARY)

        frame_denoised = denoise(frame=adapt_frame, method=cv2.MORPH_ELLIPSE, radius=denoise_val)
        contours = get_contours(frame=frame_denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            for c in contours:
                if get_contour_area(c)>8000 :
                    
                    draw_contours(frame=frame_denoised, contours=[c], color=(255, 255, 255), thickness=3)  #WHITE   

                    # hu_moments = get_hu_moments(contour=c)

                    

                    # moments = cv2.moments(c)

            
                    hu_moments = get_hu_moments(c)

                    # print("hu ->")
                    # print(hu_moments)


                    predicted_tag = clasificador.predict(hu_moments.reshape(-1, 7))

                    # hu_moments[0], hu_moments[1], hu_moments[2], hu_moments[3], hu_moments[4], hu_moments[5], hu_moments[6]

                    # print(predicted_tag)

                    if predicted_tag == 0:

                        cv2.drawContours(frame, [c],-1, (255, 0, 0), 2)#BLUE
                        x, y, w, h = cv2.boundingRect(c)
                        text = 'Square'
                        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA) 

                    elif predicted_tag == 1:
                        cv2.drawContours(frame, [c],-1, (255, 255, 0), 2)#CYAN
                        x, y, w, h = cv2.boundingRect(c)
                        text = 'Triangle'
                        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA) 
                    elif predicted_tag == 2:
                        cv2.drawContours(frame, [c],-1, (255, 0, 255), 2)#MAGENTA
                        x, y, w, h = cv2.boundingRect(c)
                        text = 'Estrella'
                        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Window', frame_denoised)
        cv2.imshow('Window2',frame )
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


main()