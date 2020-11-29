import csv
import cv2
import os
from os import listdir,makedirs
from os.path import isfile,join
import pandas as pd

path = 'Exudates'

files = [f for f in listdir(path) if isfile(join(path,f))]

print("Runnig the program .................")
n = 0
list1 = []
for image in files:
    img = cv2.imread(os.path.join(path,image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count = 0
    x, y = img.shape
    for i in range(0,x):
        for j in range(0,y):
            if(img[i][j]==0):
                count+=1
    list1.append(count)
    print(n)
    n+=1

x = pd.DataFrame(list1)
x.to_csv('Area_Exudates.csv', index=False)
