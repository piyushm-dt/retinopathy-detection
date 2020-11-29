import numpy as np
import cv2
import os

IMG_DIR = 'Normalized_Training_Set2'
IMG_DIX = 'Normalized_Testing_Set2'
x = 0
y = 0

for img in os.listdir(IMG_DIR):
    x+=1
    print(x)
    img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)
    img_array = (img_array.flatten())
    img_array  = img_array.reshape(-1, 1).T
    #print(img_array)
    with open('train.csv', 'ab') as f:
        np.savetxt(f, img_array, delimiter=",")


for img in os.listdir(IMG_DIX):
    y += 1
    print(y)
    img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)
    img_array = (img_array.flatten())
    img_array  = img_array.reshape(-1, 1).T
    #print(img_array)
    with open('test.csv', 'ab') as f:
        np.savetxt(f, img_array, delimiter=",")
