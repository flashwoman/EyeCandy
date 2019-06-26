import cv2 as cv
import numpy as np
from bookshelf.function.mouse_callback import mouse_callback, nothing
from bookshelf.function.trace_object import trace_object
from bookshelf.function.img_write import img_write


# var name rule #
# namedwindow : detect_color
# image4process : img_color


cv.namedWindow('detect_color', cv.WINDOW_NORMAL)

# img에서 찾을 sv범위(Trackbar)의 즉각적인 조절을 원할때 un-hash
# cv.createTrackbar('threshold', 'detect_color', 0, 255, nothing)
# cv.setTrackbarPos('threshold', 'detect_color', 50)

cv.setMouseCallback('detect_color', mouse_callback)

hsv = 0
lower_color1 = 0
upper_color1 = 0
lower_color2 = 0
upper_color2 = 0
lower_color3 = 0
upper_color3 = 0


while (True):


    path = "C:/flashwoman/Object-detection/image/img_book_only.png"
    img_color = cv.imread(path, cv.IMREAD_COLOR)
    # cv.imshow('original img', img_color)
    # cv.waitKey(0)

    height, width = img_color.shape[:2]
    img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)
    # BGR to HSV
    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    # *** mask 만들기 함수화 할 수 있나? *** #
    img_mask1 = cv.inRange(img_hsv, lower_color1, upper_color1)
    img_mask2 = cv.inRange(img_hsv, lower_color2, upper_color2)
    img_mask3 = cv.inRange(img_hsv, lower_color3, upper_color3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    # get rid of 'noise' _ using morphologyEx
    # kernel = cv.getStructuringElement(cv.MORPH_GRADIENT, (3,3))
    kernel = np.ones((3, 3), np.uint8)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_GRADIENT, kernel)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)
    cv.namedWindow('mask', cv.WINDOW_NORMAL)
    cv.imshow('mask', img_mask)

    # Subtract bookshelf
    img_inv_mask = cv.bitwise_not(img_mask)
    cv.namedWindow('inv_mask', cv.WINDOW_NORMAL)
    cv.imshow('inv_mask', img_inv_mask)
    img_write('bookshelf_mask', img_mask)
    img_book_only = cv.bitwise_and(img_color, img_color, mask=img_inv_mask)
    cv.namedWindow('img_book_only', cv.WINDOW_NORMAL)
    cv.imshow('img_book_only', img_book_only)
    img_bookshelf_only = cv.bitwise_and(img_color, img_color, mask=img_mask)

    # Trace Object
    # function : trace_object(img_color, img_mask)
    trace_object(img_color, img_mask)
    cv.imshow('detect_color', img_bookshelf_only)
    img_write('remove_bookshelf', img_bookshelf_only)

    # ESC == quit
    if cv.waitKey(1) & 0xFF == 27:
        break


# cv.destroyAllWindows()

import cv2 as cv
from bookshelf.function.display import display
from bookshelf.function.preprocessing import preprocessing
from bookshelf.function.contours import contours
from bookshelf.function.watershed import watershed
from bookshelf.function.img_write import img_write


'''
function_list = [
    'display', 
    'preprocessing', 
    'cv.threshold',
]
'''


# load image
path = 'C:/flashwoman/Object-detection/EyeCandy/img/bookshelf_04.jpg'
img_color = cv.imread(path)
# display('img_color', img_color)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
# display('img_gray', img_gray)

# deal with no-background colors
ret, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
# display('sep_threshold', thresh)
img_color = preprocessing(img_color, 3, 3)
img_write('1_thresh', img_color)

# deal with similar-background colors
# ret, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV)
# # display('sep_threshold', thresh)
# img_color = preprocessing(img_color)
# img_write('2_thresh', img_color)

# do contour
path = 'C:/flashwoman/Object-detection/EyeCandy/img/img_book_only.png'
img_book_only = cv.imread(path)
img_color = contours(img_book_only, img_color)
img_write('3_contour', img_color)

# do watershed
img_watershed = watershed(img_color)
# display('watershed img', img_watershed)
img_write('watershed', img_watershed)

# cv.destroyAllWindows()