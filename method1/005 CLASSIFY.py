import csv
import math
import cv2
import os
import pickle
from os import listdir,makedirs
from os.path import isfile,join
import pandas as pd
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
 
print("Running the program .................")
list1 = []

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

def calc_area(image):
    count = 0
    x, y = image.shape
    for i in range(0,x):
        for j in range(0,y):
            if(image[i][j]==255):
                count+=1
    return count

def getHomogenity(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(img, [1],[0, np.pi/2])
    h = greycoprops(glcm, 'homogeneity')[0][0]
    return h

def getEntropy(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = np.squeeze(greycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True))
    entropy = -np.sum(glcm*np.log2(glcm + (glcm==0)))
    return entropy

def Bloodvessels(image):		
	b,green_fundus,r = cv2.split(image)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	contrast_enhanced_green_fundus = clahe.apply(green_fundus)

	# applying alternate sequential filtering (3 times closing opening)
	r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
	R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
	f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
	f5 = clahe.apply(f4)		

	# removing very small contours through area parameter noise removal
	ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)	
	mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
	contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) <= 200:
			cv2.drawContours(mask, [cnt], -1, 0, -1)			
	im = cv2.bitwise_and(f5, f5, mask=mask)
	ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
	newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	

	# removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
	#vessels and also in an interval of area
	fundus_eroded = cv2.bitwise_not(newfin)	
	xmask = np.ones(image.shape[:2], dtype="uint8") * 255
	xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
	for cnt in xcontours:
		shape = "unidentified"
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   				
		if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
			shape = "circle"	
		else:
			shape = "veins"
		if(shape=="circle"):
			cv2.drawContours(xmask, [cnt], -1, 0, -1)	
	
	finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
	blood_vessels = cv2.bitwise_not(finimage)
	return blood_vessels
    
def getBloodvessels(image):
    #bv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    count = calc_area(image)
    return count

def getExudate(img):
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

    strEl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    #Dilation
    dilateImg = cv2.dilate(curImg, strEl)
    curImg = dilateImg

    #Thresholding with Complement/15
    retValue, threshImg = cv2.threshold(curImg, 235, 247, cv2.THRESH_BINARY_INV)
    curImg = threshImg

    #  Median Filtering
    medianImg = cv2.medianBlur(curImg,3)
    curImg = medianImg
    #bv = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
    count = calc_area(curImg)
    return count

def adjust_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** gamma) * 255for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def getMA(image):
    r,g,b=cv2.split(image)
    comp=255-g
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    histe=clahe.apply(comp)
    adjustImage = adjust_gamma(histe,gamma=3)
    comp = 255-adjustImage
    J =  adjust_gamma(comp,gamma=4)
    J = 255-J
    J = adjust_gamma(J,gamma=4)
    
    K=np.ones((11,11),np.float32)
    L = cv2.filter2D(J,-1,K)
    
    ret3,thresh2 = cv2.threshold(L,125,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    kernel2=np.ones((9,9),np.uint8)
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
    kernel3=np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
    count = calc_area(opening)
    return count

def getHem(img0,img1):
    #img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img3 = cv2.bitwise_not(img0, img1, mask=None)
    img4 = cv2.medianBlur(img3, 5)
    img5 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    count = calc_area(img5)
    return count

def standard_deviation_image(image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_output = clahe.apply(image)
	result = clahe_output.copy()
	i = 0
	j = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			sub_image = clahe_output[i:i+20,j:j+25]
			var = np.var(sub_image)
			result[i:i+20,j:j+25] = var
			j = j+25
		i = i+20
	return result

def deviation_from_mean(image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_output = clahe.apply(image)
	#print(clahe_output)
	result = clahe_output.copy()
	result = result.astype('int')
	i = 0
	j = 0
	while i < image.shape[0]:
		j = 0
		while j < image.shape[1]:
			sub_image = clahe_output[i:i+5,j:j+5]
			mean = np.mean(sub_image)
			sub_image = sub_image - mean
			result[i:i+5,j:j+5] = sub_image
			j = j+5
		i = i+5
	return result
'''
def get_average_intensity(green_channel):
	average_intensity = green_channel.copy()
	i = 0
	j = 0
	while i < green_channel.shape[0]:
		j = 0
		while j < green_channel.shape[1]:
			sub_image = green_channel[i:i+20,j:j+25]
			mean = np.mean(sub_image)
			average_intensity[i:i+20,j:j+25] = mean
			j = j+25
		i = i+20
	result = np.reshape(average_intensity, (average_intensity.size,1))
	return result

def get_average_hue(hue_image):
	average_hue = hue_image.copy()
	i = 0
	j = 0
	while i < hue_image.shape[0]:
		j = 0
		while j < hue_image.shape[1]:
			sub_image = hue_image[i:i+20,j:j+25]
			mean = np.mean(sub_image)
			average_hue[i:i+20,j:j+25] = mean
			j = j+25
		i = i+20
	result = np.reshape(average_hue, (average_hue.size,1))
	return result

def get_average_saturation(hue_image):
	average_hue = hue_image.copy()
	i = 0
	j = 0
	while i < hue_image.shape[0]:
		j = 0
		while j < hue_image.shape[1]:
			sub_image = hue_image[i:i+20,j:j+25]
			mean = np.mean(sub_image)
			average_hue[i:i+20,j:j+25] = mean
			j = j+25
		i = i+20
	result = np.reshape(average_hue, (average_hue.size,1))
	return result
'''

def get_SD_data(sd_image):	
	feature = np.reshape(sd_image, (sd_image.size,1))
	return feature

def get_HUE_data(hue_image):	
	feature = np.reshape(hue_image,(hue_image.size,1))	
	return feature

def get_saturation_data(s_image):
	feature = np.reshape(s_image,(s_image.size,1))	
	return feature


def get_INTENSITY_data(intensity_image):	
	feature = np.reshape(intensity_image,(intensity_image.size,1))	
	return feature

def get_EDGE_data(edge_candidates_image):
	feature_4 = np.reshape(edge_candidates_image,(edge_candidates_image.size,1))	
	return feature_4

def get_RED_data(red_channel):	
	feature = np.reshape(red_channel, (red_channel.size,1))	
	return feature

def get_GREEN_data(green_channel):
	feature = np.reshape(green_channel, (green_channel.size,1))	
	return feature


def getMultiple(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    '''
    do something
    '''
    b,g,r = cv2.split(img)
    h,s,v = cv2.split(hsv_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    contrast_enhanced_green = clahe.apply(g)

    #average_intensity = get_average_intensity(contrast_enhanced_green)/255
    #average_hue = get_average_hue(h)/255
    #average_saturation = get_average_saturation(s)/255
    var = standard_deviation_image(contrast_enhanced)
    dev = deviation_from_mean(gray)

    feature1 = np.mean(get_SD_data(var)/255)
    feature2 = np.mean(get_HUE_data(h)/255)
    feature3 = np.mean(get_saturation_data(s)/255)
    feature4 = np.mean(get_INTENSITY_data(contrast_enhanced)/255)
    feature5 = np.mean(get_RED_data(r)/255)
    feature6 = np.mean(get_GREEN_data(g)/255)
    feature7 = np.mean(get_HUE_data(dev)/255)

    list1.append(feature1)
    list1.append(feature2)
    list1.append(feature3)
    list1.append(feature4)
    list1.append(feature5)
    list1.append(feature6)
    list1.append(feature7)

image = cv2.imread('unity.jpg')
img = color_normalize(image)

list1.append(getMA(img))
list1.append(getExudate(img))
one = Bloodvessels(img)
list1.append(getBloodvessels(one))
list1.append(getHem(img, one))
list1.append(getEntropy(img))
list1.append(getHomogenity(img))
getMultiple(img)
print('Features of image extracted: ', list1)

arr = list1
arr.sort(reverse=True)
brr = []
for i in range(9):
  brr.append(arr[i])
list2 = np.reshape(brr, (-1, 9))

#print('Features of image to be used: ', list2)

print('Diabetic Retinopathy Grade: ')

file1 = 'file1.sav'
model1 = pickle.load(open(file1, 'rb'))
print('Using random forest classifier: ', (model1.predict(list2)))
file2 = 'file2.sav'
model2 = pickle.load(open(file2, 'rb'))
print('Using support vector classifier: ', (model2.predict(list2)))

print('Macular Edema Grade: ')

file3 = 'file3.sav'
model1 = pickle.load(open(file3, 'rb'))
print('Using random forest classifier: ', (model1.predict(list2)))
file4 = 'file4.sav'
model2 = pickle.load(open(file4, 'rb'))
print('Using support vector classifier: ', (model2.predict(list2)))
