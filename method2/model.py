import os, time, warnings
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings(action="ignore")
df = pd.read_csv('mergedDataset.csv')
af = pd.read_csv('mergedLabels.csv')


label1 = af.iloc[:, [0]]
label2 = af.iloc[:, [1]]

featureX = StandardScaler().fit_transform(df)

trainX, testX, trainY, testY = train_test_split(featureX, label1, test_size=0.3)
trainX, testX, trainZ, testZ = train_test_split(featureX, label2, test_size=0.3)

print("Principal Component Analysis ")
pca = PCA(n_components=100,iterated_power='auto', random_state=21,svd_solver='auto' , whiten=True)

features_pca = pca.fit_transform(trainX)
features_pca_ = pca.fit_transform(testX)
# features_pca_ = pca.fit_transform(features_)
print("Original number of features:", trainX.shape[1])
print("Reduced number of features:", features_pca.shape[1])

X_Train = features_pca
X_Test = features_pca_
Y_Train = trainY
Y_Test = testY
Z_Train = trainZ
Z_Test = testZ

print('\nFor Retinopthy grades: \n')

print("\nRandom Forest\n")
trainedforest = RandomForestClassifier(n_estimators=features_pca.shape[1],criterion='entropy', max_depth=4, max_features='auto').fit(X_Train, Y_Train)
predictions = trainedforest.predict(X_Test)
print(confusion_matrix(Y_Test, predictions))
print(classification_report(Y_Test, predictions))

print("\nSupport Vector Classifier\n")
svc = SVC(kernel='rbf', C=1)
print('\nFor Retinopthy grades: \n')
model1 = svc.fit(X_Train, Y_Train)
predictions = model1.predict(X_Test)
print(confusion_matrix(Y_Test, predictions))
print(classification_report(Y_Test, predictions))


print('\nFor risk of macular edema: \n')

print("\nRandom Forest\n")
trainedforest2 = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='entropy', max_features='auto').fit(X_Train, Z_Train)
predictions2 = trainedforest2.predict(X_Test)
print(confusion_matrix(Z_Test, predictions2))
print(classification_report(Z_Test, predictions2))

print("\nSupport Vector Classifier\n")
model2 = svc.fit(X_Train, Z_Train)
predictions = model2.predict(X_Test)
print(confusion_matrix(Z_Test, predictions))
print(classification_report(Z_Test, predictions))

file1 = 'file1.sav'
pickle.dump(trainedforest, open(file1, 'wb'))
file2 = 'file2.sav'
pickle.dump(trainedforest2, open(file2, 'wb'))

file3 = 'file3.sav'
pickle.dump(model1, open(file3, 'wb'))
file4 = 'file4.sav'
pickle.dump(model2, open(file4, 'wb'))
