import os
from os import listdir,makedirs
from os.path import isfile,join
import shutil
import glob
import argparse
from tqdm import tqdm

import matplotlib as mpl
mpl.use('TkAgg')

import cv2
import matplotlib.pyplot as plt
import numpy as np

path = 'Training_Set' # Source Folder
dstpath = 'Normalized_Training_Set' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in same folder")
    
# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))]


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
  img = cv2.resize(img, (512, 512)) 
  img = (img * 255).astype(np.uint8)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

for image in files:
        img = cv2.imread(os.path.join(path,image))
        img = color_normalize(img)
        dstPath = join(dstpath,image)
        cv2.imwrite(dstPath,img)

