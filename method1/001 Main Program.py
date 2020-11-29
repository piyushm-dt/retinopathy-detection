import pandas as pd
import numpy as np
import warnings
import pickle
import IPython
from IPython import display
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
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
model = PCA(n_components=0.98)
model.fit(X_Train)
model.fit(X_Test)
PCA(copy=True, iterated_power='auto', n_components=3, random_state=21,svd_solver='auto', tol=0.0, whiten=False)
X_3D = model.transform(X_Train)
X_3D_ = model.transform(X_Test)


print("About PCA: ")
print(model.explained_variance_)
print(model.explained_variance_ratio_)
#################################################################################################################################
print('\n Retinopathy grade: \n')

print("\nRandom Forest\n")
trainedforest = RandomForestClassifier(n_estimators=100).fit(X_3D, Y_Train)
predictions = trainedforest.predict(X_3D_)
print(confusion_matrix(Y_Test, predictions))#,labels=np.unique(predictions) ))
print(classification_report(Y_Test, predictions))#,labels=np.unique(predictions) ))

print("\nSupport Vector Classifier\n")
svc = SVC(kernel='poly', C=2)
model1 = svc.fit(X_3D, Y_Train)
predictions = model1.predict(X_3D_)
print(confusion_matrix(Y_Test, predictions))#, labels=np.unique(predictions)))
print(classification_report(Y_Test, predictions))#, labels=np.unique(predictions)))
###################################################################################################################################
print('Risk of macular edema: \n')

X_Train_, X_Test_, Y_Train_, Y_Test_ = train_test_split(features, label2, test_size=0.3, random_state=42)
model.fit(X_Train_)
model.fit(X_Test_)
PCA(copy=True, iterated_power='auto', n_components=3, random_state=21,svd_solver='auto', tol=0.0, whiten=False)
Y_3D = model.transform(X_Train_)
Y_3D_ = model.transform(X_Test_)


print("\nRandom Forest\n")
trainedforest2 = RandomForestClassifier(n_estimators=100).fit(Y_3D, Y_Train_)
predictions2 = trainedforest2.predict(Y_3D_)
print(confusion_matrix(Y_Test_, predictions2))
print(classification_report(Y_Test_, predictions2))

print("\nSupport Vector Classifier\n")
model2 = svc.fit(Y_3D, Y_Train_)
predictions2 = model2.predict(Y_3D_)
print(confusion_matrix(Y_Test_, predictions2))
print(classification_report(Y_Test_, predictions2))


filename1 = 'file1.sav'
pickle.dump(trainedforest, open(filename1, 'wb'))

filename2 = 'file2.sav'
pickle.dump(trainedforest2, open(filename2, 'wb'))


filename3 = 'file3.sav'
pickle.dump(model1, open(filename3, 'wb'))

filename4 = 'file4.sav'
pickle.dump(model2, open(filename4, 'wb'))

