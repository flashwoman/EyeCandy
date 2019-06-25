import cv2 as cv
from bookshelf.function.contours import contours
from bookshelf.function.preprocessing import preprocessing


# 1. load image
img = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/img_book_only.png')
img_org = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/bookshelf_04.jpg')

# 2. preprocess images
img = preprocessing(img)
img, rect = contours(img,img_org)       # rect에 좌표 저장 [Left_Top, Right_Bottom]
# organizing rect
lt_x=[]; lt_y=[]; rb_x=[]; rb_y=[]; lt_coord=[]; rb_coord=[]
for i in range(len(rect)):
    lt_x.append(list(rect[i][0])[0])
    lt_y.append(list(rect[i][0])[1])
    rb_x.append(list(rect[i][1])[0])
    rb_y.append(list(rect[i][1])[1])
    lt_coord.append([lt_x,lt_y])
    rb_coord.append([rb_x,rb_y])



## 6. Rearrange Books
from bookshelf.function.create_bookshelf import create_bookshelf

# 1.
coords = create_bookshelf(img_org)
print(coords)

