import pandas as pd
import IPython
from IPython import display
import numpy as np
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize

df = pd.read_csv('ff_train.csv')

label1 = df['Retinopathy grade']
label2 = df['Risk of macular edema']
features = df.drop(['Image name','Retinopathy grade','Risk of macular edema'], axis=1)

model = PCA(n_components=3)
model.fit(features)
PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,svd_solver='auto', tol=0.0, whiten=False)
X_3D = model.transform(features)

a_x, a_y, a_z = X_3D[label1 == 0].T
b_x, b_y, b_z = X_3D[label1 == 1].T
c_x, c_y, c_z = X_3D[label1 == 2].T
d_x, d_y, d_z = X_3D[label1 == 3].T
e_x, e_y, e_z = X_3D[label1 == 4].T

f_x, f_y, f_z = X_3D[label2 == 0].T
g_x, g_y, g_z = X_3D[label2 == 1].T
h_x, h_y, h_z = X_3D[label2 == 2].T

a = go.Scatter3d(
    x=a_x,
    y=a_y,
    z=a_z,
    mode='markers',
    name='Grade0',
    marker=dict(
        color='blue',
        size=3,
        opacity=0.9
    )
)

b = go.Scatter3d(
    x=b_x,
    y=b_y,
    z=b_z,
    mode='markers',
    name='Grade1',
    marker=dict(
        color='red',
        size=3,
        opacity=0.9
    )
)

c = go.Scatter3d(
    x=c_x,
    y=c_y,
    z=c_z,
    mode='markers',
    name='Grade2',
    marker=dict(
        color='yellow',
        size=3,
        opacity=0.9
    )
)

d = go.Scatter3d(
    x=d_x,
    y=d_y,
    z=d_z,
    mode='markers',
    name='Grade3',
    marker=dict(
        color='green',
        size=3,
        opacity=0.9
    )
)

e = go.Scatter3d(
    x=e_x,
    y=e_y,
    z=e_z,
    mode='markers',
    name='Grade4',
    marker=dict(
        color='magenta',
        size=3,
        opacity=0.9
    )
)

f = go.Scatter3d(
    x=f_x,
    y=f_y,
    z=f_z,
    mode='markers',
    name='Grade0',
    marker=dict(
        color='red',
        size=3,
        opacity=0.9
    )
)
g = go.Scatter3d(
    x=g_x,
    y=g_y,
    z=g_z,
    mode='markers',
    name='Grade1',
    marker=dict(
        color='blue',
        size=3,
        opacity=0.9
    )
)
h = go.Scatter3d(
    x=h_x,
    y=h_y,
    z=h_z,
    mode='markers',
    name='Grade2',
    marker=dict(
        color='green',
        size=3,
        opacity=0.9
    )
)



layout1 = go.Layout(title = 'Retinopathy')
fig1 = go.Figure(data=[a, b, c, d, e], layout=layout1)
iplot(fig1)

layout2 = go.Layout(title = 'Macular Edema')
fig2 = go.Figure(data=[f, g, h], layout=layout2)
iplot(fig2)

