# Reference Page of SOM clustering exaples
#  : https://github.com/annoviko/pyclustering/blob/master/pyclustering/nnet/examples/som_examples.py
#  : https://github.com/annoviko/pyclustering/blob/master/pyclustering/nnet/som.py
# Reference Image
#  : https://github.com/annoviko/pyclustering/blob/master/docs/img/target_som_processing.png

import random
import cv2 as cv
from bookshelf.function.contours import contours
from bookshelf.function.preprocessing import preprocessing
from bookshelf.function.display import display
from pyclustering.nnet.som import som, type_conn, type_init, som_parameters


# 1. load image
img = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/img_book_only.png')
img_org = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/bookshelf_04.jpg')

# 2. preprocess images
img = preprocessing(img)
img, rect = contours(img,img_org)

# 3. bring mid coordinates
coord = []
for i in range(len(rect)):
    # extracting colouor by using mid-coordinates of the book
    x = (list(rect[i][0])[0] + list(rect[i][1])[0])/2
    y = (list(rect[i][0])[1] + list(rect[i][1])[1])/2
    coord.append([x, y])

print('book_mid_coordinates :', coord)

# 4. get 'bgr' from coordinates
bgr_val = []
for i in range(len(coord)):
    ## 1. x, y 좌표값 받아오기
    x, y = coord[i]
    # print(len(coord), i, coord[i], a, img_org.item(round(b), round(a), 0))
    ## 2. int값
    x = round(x)
    y = round(y)
    ## .item(y, x, (0=b,1=g,2=r))
    bgr_val.append([img_org.item(y, x, 0),
                    img_org.item(y, x, 1),
                    img_org.item(y, x, 2)])

print('selected_pixel_bgr :', bgr_val)

# save to csv
# csvData = rect
# with open('C:/flashwoman/Object-detection/testfiles_sey/books_coord.csv', 'w') as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerows(csvData)
# csvFile.close()


## SOM
# read sample 'Lsun' from file
# sample = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)
# [[2.0, 3.0], [0.387577, 0.268546], [0.17678, 0.582963], [3.277701, 0.814082], [0.387577, 0.17678], ...]
sample = bgr_val

# create SOM parameters
parameters = som_parameters()

# create self-organized feature map with size 7x7
rows = 6  # five rows
cols = 6  # five columns
structure = type_conn.grid_four  # each neuron has max. four neighbors.
network = som(rows, cols, structure, parameters)

# train network on 'Lsun' sample during 100 epochs.
network.train(sample, 50)

# simulate trained network using randomly modified point from input dataset.
index_point = random.randint(0, len(sample) - 1)
point = sample[index_point]  # obtain randomly point from data
point[0] += random.random() * 0.2  # change randomly X-coordinate
point[1] += random.random() * 0.2  # change randomly Y-coordinate
index_winner = network.simulate(point)

# check what are objects from input data are much close to randomly modified.
index_similar_objects = network.capture_objects[index_winner]

# neuron contains information of encoded objects
print("Point '%s' is similar to objects with indexes '%s'." % (str(point), str(index_similar_objects)))
print("Coordinates of similar objects:")
for index in index_similar_objects: print("\tPoint:", sample[index])

# result visualization:
# show distance matrix (U-matrix).
network.show_distance_matrix()
# show density matrix (P-matrix).
network.show_density_matrix()
# show winner matrix.
network.show_winner_matrix()
# show self-organized map.
network.show_network()



cv.destroyAllWindows()


