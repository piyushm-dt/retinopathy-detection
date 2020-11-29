import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings(action="ignore")

df = pd.read_csv('main.csv')

label1 = df['Retinopathy grade']
label2 = df['Risk of macular edema']
features = df.drop(['Image name','Retinopathy grade','Risk of macular edema'], axis=1)

features = StandardScaler().fit_transform(features)
label1 = LabelEncoder().fit_transform(label1)
label2 = LabelEncoder().fit_transform(label2)
#################################################################################################################################
X_Train, X_Test, Y_Train, Y_Test = train_test_split(features, label1, test_size=0.3, random_state=42)

print('\n Retinopathy grade: \n')

print("\nRandom Forest\n")
trainedforest = RandomForestClassifier(n_estimators=100).fit(X_Train, Y_Train)
predictions = trainedforest.predict(X_Test)
print(confusion_matrix(Y_Test, predictions))#,labels=np.unique(predictions) ))
print(classification_report(Y_Test, predictions))#,labels=np.unique(predictions) ))

print("\nSupport Vector Classifier\n")
svc = SVC(kernel='poly', C=2)
model1 = svc.fit(X_Train, Y_Train)
predictions = model1.predict(X_Test)
print(confusion_matrix(Y_Test, predictions))#, labels=np.unique(predictions)))
print(classification_report(Y_Test, predictions))#, labels=np.unique(predictions)))
###################################################################################################################################
X_Train_, X_Test_, Y_Train_, Y_Test_ = train_test_split(features, label2, test_size=0.3, random_state=42)

print('Risk of macular edema: \n')

print("\nRandom Forest\n")
trainedforest2 = RandomForestClassifier(n_estimators=100).fit(X_Train_, Y_Train_)
predictions2 = trainedforest2.predict(X_Test_)
print(confusion_matrix(Y_Test_, predictions2))
print(classification_report(Y_Test_, predictions2))

print("\nSupport Vector Classifier\n")
model2 = svc.fit(X_Train_, Y_Train_)
predictions2 = model2.predict(X_Test_)
print(confusion_matrix(Y_Test_, predictions2))
print(classification_report(Y_Test_, predictions2))
