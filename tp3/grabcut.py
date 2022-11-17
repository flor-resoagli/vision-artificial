import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def takeImg():
    cam = cv.VideoCapture(0)
    cv.namedWindow("Webcam")

    while True:
        ret, frame = cam.read()

        if not ret:
            print('failed to grab frame')
            return -1
            break

        cv.imshow('test', frame)

        k = cv.waitKey(1)

        if k%256 == 27:
            return -1
            break

        if k%256 == 32:
            cv.imwrite("img.jpg", frame)
            break
    
    return -1

takeImg()
img = cv.imread('img.jpg')

# si usamos el metodo de GC_INIT_WITH_RECT no es necesario camara por eso hacemos una matriz de 0
mask = np.zeros(img.shape[:2], np.uint8)

# These are arrays used by the algorithm internally. You just create two np.float64 type zero arrays
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# usamos roi para agarrar el rect
rect = cv.selectROI("img", img, fromCenter=False, showCrosshair=True)
print(mask)

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)

print(mask)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

print(mask2)


img = img*mask2[:, :, np.newaxis]

plt.subplot(121)
plt.imshow(img)
plt.title("Grabcut")
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.imshow(cv.cvtColor(cv.imread("img.jpeg"), cv.COLOR_BGR2RGB))
plt.title("Original")
plt.xticks([]), plt.yticks([]) 
plt.show() 

# cv.imshow("img", img)
# cv.waitKey()   

