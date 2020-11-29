import os, time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

df1 = pd.read_csv('train.csv')
file2 = pd.read_csv('trainLabels.csv')
df2 = pd.read_csv('test.csv')
file4 = pd.read_csv('testLabels.csv')

label1 = file2['Retinopathy grade']
label2 = file2['Risk of macular edema']
label3 = file4['Retinopathy grade']
label4 = file4['Risk of macular edema']

features = StandardScaler().fit_transform(df1)
features_ = StandardScaler().fit_transform(df2)

print("Principal Component Analysis ")
list1 = [10,20,25,50,75,100]
for x in list1:
    print(x)
    print('..........\n')
    pca = PCA(n_components=x, whiten=True)
    features_pca = pca.fit_transform(features)
    features_pca_ = pca.fit_transform(features_)

    print("For training set: ")
    print("Original number of features:", features.shape[1])
    print("Reduced number of features:", features_pca.shape[1])

    print("For testing set: ")
    print("Original number of features:", features_.shape[1])
    print("Reduced number of features:", features_pca_.shape[1])

    X_Train = features_pca
    X_Test = features_pca_
    Y_Train = label1
    Y_Test = label3
    Z_Train = label2
    Z_Test = label4

    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators':[100,200,500],
        'max_features':['auto','sqrt','log2'],
        'max_depth':[4,5,6],
        'criterion':['gini','entropy']
        }
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=4)
    CV_rfc.fit(X_Train, Y_Train)
    print(CV_rfc.best_params_)

    svc = SVC()
    param_grid = {
        'C':[1,5,10],
        'kernel':['linear','rbf','poly'],
        'random_state':[0,21,42]
        }
    CV_svc = GridSearchCV(estimator=svc, param_grid=param_grid, cv=4)
    CV_svc.fit(X_Train, Y_Train)
    print(CV_svc.best_params_)
