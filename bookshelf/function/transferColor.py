# https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/
# using nothing but the mean and standard flashwomaniation of the image channels (Lab)

# pip install color_transfer
from color_transfer import color_transfer
import cv2 as cv
import numpy as np


def display(winname, img):
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)

# load the images
source = 'C:/flashwoman/Object-detection/image/bookshelf_04.jpg'
target = 'C:/flashwoman/Object-detection/image/emptybookshelf_box.jpg'

# color_transfer
source = cv.imread(source, cv.COLOR_BAYER_BG2BGRA)
target = cv.imread(target)
transfer = color_transfer(source, target)

# get height, width
height, width = target.shape[:2]
print(width,height)

# box 쌓기
boxes = [] 
coords = [] 
for i in range(0, 4 ):
	boxes.append(transfer)

res = np.vstack(boxes)
path = "C:/flashwoman/Object-detection/image/bookshelf_fin.jpg"
cv.imwrite(path, res)
boxes = cv.imread("C:/flashwoman/Object-detection/image/bookshelf_fin.jpg")

# display("Source", source)
# display("Target", target)
# display("Transfer", transfer)
# display("boxes", boxes)

for i in range(0, 4):
	h_w = height * 1.01 * i
	x = int(width * 0.0157)
	y = int(height * 0.93 + h_w )
	coord = (x, y)
	coords.append(  coord  )
	cv.circle(boxes, coord, 2, (0, 0, 255), -1)


display("boxes2", boxes)
cv.waitKey(0)