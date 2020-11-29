import numpy as np
import csv
import cv2
import os,shutil
from os import listdir,makedirs
from os.path import isfile,join

xpath = 'BloodVessels' # Source Folder
path = 'Normalized_Training_Set'
dstpath = 'Hemorrhages' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in same folder")


# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))]
xfiles = [f for f in listdir(path) if isfile (join(xpath,f))]

for image1, image2 in zip(files, xfiles):
        img0 = cv2.imread(os.path.join(path,image1))
        img1 = cv2.imread(os.path.join(xpath,image2))
        img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img3 = cv2.bitwise_not(img0, img2, mask=None)
        #img3 = cv2.bitwise_and(img2, img3, mask=None)
        img4 = cv2.medianBlur(img3, 5)
        img3 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        dstPath = join(dstpath,image1)
        cv2.imwrite(dstPath,img3)
