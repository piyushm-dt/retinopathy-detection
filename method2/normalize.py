import os
from os import listdir,makedirs
from os.path import isfile,join
import shutil
import glob
import cv2
import numpy as np

path1 = 'Training_Set' # Source Folder
dstpath1 = 'Normalized_Training_Set2' # Destination Folder
path2 = 'Testing_Set' # Source Folder
dstpath2 = 'Normalized_Testing_Set2' # Destination Folder

try:
    makedirs(dstpath1)
except:
    print ("Directory already exist, images will be written in same folder")

try:
    makedirs(dstpath2)
except:
    print ("Directory already exist, images will be written in same folder")
    
# Folder won't used
file1 = [f for f in listdir(path1) if isfile(join(path1,f))]
file2 = [f for f in listdir(path2) if isfile(join(path2,f))]


def color_normalize(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  b = img.copy()
  # set green and red channels to 0
  b[:, :, 1] = 0
  b[:, :, 2] = 0

  g = img.copy()
  # set blue and red channels to 0
  g[:, :, 0] = 0
  g[:, :, 2] = 0

  r = img.copy()
  # set blue and green channels to 0
  r[:, :, 0] = 0
  r[:, :, 1] = 0

  img = (b/b.max() + g/g.max() + r/r.max())
  img = cv2.resize(img, (256, 256)) 
  img = (img * 255).astype(np.uint8)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img
x=0
for image in file1:
    print(x)
    x+=1
    img = cv2.imread(os.path.join(path1,image))
    img = color_normalize(img)
    dstPath = join(dstpath1,image)
    cv2.imwrite(dstPath,img)

x=0
for image in file2:
    print(x)
    x+=1
    img = cv2.imread(os.path.join(path2,image))
    img = color_normalize(img)
    dstPath = join(dstpath2,image)
    cv2.imwrite(dstPath,img)


