import cv2 as cv


# show webcam
cap = cv.VideoCapture(0)
alpha_slider_max = 100

def grayscale(img): 
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def binary(val):
    ret, img = cap.read()
    img = cv.flip(img, 1)
    g = grayscale(img)
    ret1, thresh1 = cv.threshold(g, val, 255, cv.THRESH_BINARY)
    cv.imshow("Binary", thresh1)

while True :

    # shows video
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    # monochrome capture
    gray = grayscale(frame)
    
    cv.imshow('WebCapture', gray)

    # show binary threshold and trackbar
    cv.namedWindow('Binary')
    cv.createTrackbar('Trackbar', 'Binary', 0, alpha_slider_max, binary)
    
    binary(0)

    # stops with Z key
    if cv.waitKey(1) == ord('z'):
        break

    


# stops camara and closes open windows
cap.release()
cv.destroyAllWindows()