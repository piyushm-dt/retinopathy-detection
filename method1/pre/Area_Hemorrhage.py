import csv
import cv2
import os
from os import listdir,makedirs
from os.path import isfile,join
import pandas as pd

path = 'Hemorrhages'

files = [f for f in listdir(path) if isfile(join(path,f))]

def calc_area(image):
    count = 0
    x, y = img.shape
    for i in range(0,x):
        for j in range(0,y):
            if(img[i][j]==255):
                count+=1
    return count
  
print("Running the program .................")
list1 = []
x = 1
for image in files:
    img = cv2.imread(os.path.join(path,image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count = calc_area(img)
    print(x)
    print("......... \n")
    x+=1
    list1.append(count)

x = pd.DataFrame(list1)
x.to_csv('Area_Hemorrhage.csv', index=False)
