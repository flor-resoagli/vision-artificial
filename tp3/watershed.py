
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from skimage.segmentation import clear_border

img = cv2.imread('coins.webp')
# gray = img[:,:, 0] #Image equivalent to grey image.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# noise removal
# Morphological operations to remove small noise - opening
# Remove holes - closings
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
opening = clear_border(opening) #Remove edge touching grains
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
#pixels_to_um = 0.5 # 1 pixel = 454 nm (got this from the metadata of original image)


#Now we know that the regions at the center of img is for sure img
sure_bg = cv2.dilate(closing,kernel,iterations=5)

# Finding sure foreground area using distance transform and thresholding
dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,5)


# #Let us threshold the dist transform by starting at 1/2 its max value.
# #print(dist_transform.max()) gives about 21.9
ret2, sure_fg = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)

# #Later you realize that 0.25* max value will not separate the img well.
# #High value like 0.7 will not recognize some img. 0.5 seems to be a good compromize

# # Unknown ambiguous region is nothing but bkground - foreground
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# #Now we create a marker and label the regions inside. 
# # For sure regions, both foreground and background will be labeled with positive numbers.
# # Unknown regions will be labeled 0. 
_, markers = cv2.connectedComponents(sure_fg)
print(markers)
print("------------------")

# #One problem rightnow is that the entire background pixels is given value 0.
# #This means watershed considers this region as unknown.
# #So let us add 10 to all labels so that sure background is not 0, but 10
markers = markers+1

# # Now, mark the region of unknown with zero
markers[unknown==255] = 0


markers = cv2.watershed(img,markers)
# #The boundary region will be marked -1

# color boundaries in yellow. 
img[markers == -1] = [0,255,255]  

img[markers == -1] = [0,255,255]  

img2 = color.label2rgb(markers, bg_label=0)

cv2.imshow('sure_bg', sure_bg)
cv2.imshow('sure_fg', sure_fg)
cv2.imshow('Overlay on original image', img)
cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)

#Now, time to extract properties of detected img
# regionprops function in skimage measure module calculates useful parameters for each object.
# regions = measure.regionprops(markers, intensity_image=img)

#Can print various parameters for all objects
# for prop in regions:
#     print('Label: {} Area: {}'.format(prop.label, prop.area))


 
    

