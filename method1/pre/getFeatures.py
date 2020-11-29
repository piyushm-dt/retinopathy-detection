import csv
import cv2
import os
from os import listdir,makedirs
from os.path import isfile,join
import pandas as pd
import numpy as np
import math

path = 'Normalized_Training_Set'

files = [f for f in listdir(path) if isfile(join(path,f))]

# define functions -------------
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


def get_SD_data(sd_image):	
	feature = np.reshape(sd_image, (sd_image.size,1))
	return feature

def get_HUE_data(hue_image):	
	feature = np.reshape(hue_image,(hue_image.size,1))	
	return feature

def get_saturation_data(s_image):
	feature = np.reshape(s_image,(s.size,1))	
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


print("Running the program .................")
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []

x = 1
for image in files:
    img = cv2.imread(os.path.join(path,image))
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
    list2.append(feature2)
    list3.append(feature3)
    list4.append(feature4)
    list5.append(feature5)
    list6.append(feature6)
    list7.append(feature7)

    print(x)
    print('......\n')
    x+=1

#'''
a = pd.DataFrame(list1)
a.to_csv('f1.csv', index=False)
b = pd.DataFrame(list2)
b.to_csv('f2.csv', index=False)
c = pd.DataFrame(list3)
c.to_csv('f3.csv', index=False)
d = pd.DataFrame(list4)
d.to_csv('f4.csv', index=False)
e = pd.DataFrame(list5)
e.to_csv('f5.csv', index=False)
f = pd.DataFrame(list6)
f.to_csv('f6.csv', index=False)
g = pd.DataFrame(list7)
g.to_csv('f7.csv', index=False)
#'''
