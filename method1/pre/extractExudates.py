import cv2
import numpy as np
import os,glob
from os import listdir,makedirs
from os.path import isfile,join

path = 'Normalized_Training_Set'
dstpath = 'Exudates'

try:
        makedirs(dstpath)

except:
        print('Directory exists')

files = [f for f in listdir(path) if isfile(join(path,f))]

def exudate(img):

  jpegImg = 0
  grayImg = 0
  curImg = 0

  jpegImg = img
  curImg = np.array(img)    ##Convert jpegFile to numpy array (Required for CV2)

  gcImg = curImg[:,:,1]
  curImg = gcImg

  clahe = cv2.createCLAHE()
  clImg = clahe.apply(curImg)
  curImg = clImg

  #Creating Structurig Element
  strEl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
  #Dilation
  dilateImg = cv2.dilate(curImg, strEl)
  curImg = dilateImg

          #Thresholding with Complement/15
  retValue, threshImg = cv2.threshold(curImg, 235, 247, cv2.THRESH_BINARY_INV)
  curImg = threshImg

  #Median Filtering
  medianImg = cv2.medianBlur(curImg,3)
  curImg = medianImg
  return curImg
  #plt.imshow(cv2.bitwise_and(img, img, mask = curImg))


for image in files:
        img = cv2.imread(os.path.join(path,image))
        ex=exudate(img)
        dstPath = join(dstpath, image)
        cv2.imwrite(dstPath, ex)
