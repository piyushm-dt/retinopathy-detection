import csv
import math
import cv2
import os
from os import listdir,makedirs
from os.path import isfile,join
import pandas as pd
import numpy as np
from skimage.feature import greycomatrix

path = 'Normalized_Training_Set'

files = [f for f in listdir(path) if isfile(join(path,f))]
  
print("Running the program .................")
x = 0
list1 = []
for image in files:
    img = cv2.imread(os.path.join(path,image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = np.squeeze(greycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True))
    entropy = -np.sum(glcm*np.log2(glcm + (glcm==0)))
    list1.append(entropy)

x = pd.DataFrame(list1)
x.to_csv('entropy.csv', index=False)
