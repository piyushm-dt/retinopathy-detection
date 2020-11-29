import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

#---------------------------------------------------------------------------------
df = pd.read_csv('ff_train.csv')
#ef = pd.read_csv('ff_test.csv')

label1 = df['Retinopathy grade']
label2 = df['Risk of macular edema']
features = df.drop(['Image name','Retinopathy grade','Risk of macular edema'], axis=1)

#features_ = ef.drop(['Image name','Retinopathy grade','Risk of macular edema'], axis=1)

features = StandardScaler().fit_transform(features)
#features_= StandardScaler().fit_transform(features_)
label1 = LabelEncoder().fit_transform(label1)
label2 = LabelEncoder().fit_transform(label2)
#-------------------------------------------------------------------------------------
X_Train, X_Test, Y_Train, Y_Test = train_test_split(features, label1, test_size=0.30, random_state=101)
start = time.process_time()
trainedforest = RandomForestClassifier(n_estimators=200).fit(X_Train, Y_Train)
print(time.process_time() - start)
predictions = trainedforest.predict(X_Test)
print(confusion_matrix(Y_Test, predictions))
print(classification_report(Y_Test, predictions))

#forest(features, label1)
#forest(features, label2)
# ------------------------------------------------------------------------------------

pca = PCA(n_components=3, svd_solver='full')
X_pca = pca.fit_transform(features)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2','PC3'])
PCA_df = pd.concat([PCA_df, df['Retinopathy grade'], df['Risk of macular edema']], axis = 1)
PCA_df['Retinopathy grade'] = LabelEncoder().fit_transform(PCA_df['Retinopathy grade'])
PCA_df['Risk of macular edema'] = LabelEncoder().fit_transform(PCA_df['Risk of macular edema'])
print(PCA_df.head())

print(pca.explained_variance_)

#--------------------------------------------------------------------------------------

figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
classes = [0, 1, 2, 3, 4]
colors = ['r', 'b', 'g', 'y', 'm']

for clas, color in zip(classes, colors):
    plt.scatter(PCA_df.loc[PCA_df['Retinopathy grade'] == clas, 'PC1'], PCA_df.loc[PCA_df['Retinopathy grade'] == clas, 'PC2'], PCA_df.loc[PCA_df['Retinopathy grade'] == clas, 'PC3'], c = color)

plt.title('Diabetic Retinopathy', fontsize = 15)
plt.legend(['Grade0','Grade1','Grade2','Grade3','Grade4'])
plt.grid()
plt.show()

for clas, color in zip(classes, colors):
    plt.scatter(PCA_df.loc[PCA_df['Risk of macular edema'] == clas, 'PC1'], PCA_df.loc[PCA_df['Risk of macular edema'] == clas, 'PC2'],PCA_df.loc[PCA_df['Risk of macular edema'] == clas, 'PC3'], c = color)

plt.title('Macular Edema', fontsize = 15)
plt.legend(['Grade0','Grade1','Grade2'])
plt.grid()
plt.show()

#-------------------------------------------------------------------------------------
