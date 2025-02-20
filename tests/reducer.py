import torch.utils
import path_setup
from DeepUtilsJ.losses import VAELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from DeepUtilsJ.models.vaeReducer import VaeReducer
from DeepUtilsJ.training import Trainer
from DeepUtilsJ.losses import VAELoss
from torch.utils.data import Dataset, DataLoader

import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer, load_digits
import plotly.express as px


# X, y = load_iris(return_X_y=True)
# X, y = load_breast_cancer(return_X_y=True)
X, y = load_digits(return_X_y=True)



reducer = VaeReducer(input_dim=X.shape[-1], latent_dim=2, encoder_layers=[150]*2, decoder_layers=[150]*2)
reducer.to_gpu()
reducer.fit(X, max_epochs=1500, target_r2=0.9, beta=0.2)
Z = reducer.transform(X)

fig = px.scatter()
fig.add_scatter(x=Z[:,0], y=Z[:,1], mode='markers', marker=dict(color=y, size=5, )) #colorscale='rainbow'
fig.update_layout(title="2D representation", plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"),
                                                     legend=dict(bordercolor='white', borderwidth=1))
fig.show()

from sklearn.manifold import Isomap

oIsomap  = Isomap(n_neighbors=6, n_components=2, metric='l2').fit(X)
mZ       = oIsomap.transform(X)
fig = px.scatter()
fig.add_scatter(x=mZ[:,0], y=mZ[:,1], mode='markers', marker=dict(color=y, size=5, )) #colorscale='rainbow'
fig.update_layout(title="2D representation", plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"),
                                                     legend=dict(bordercolor='white', borderwidth=1))
fig.show()


print('done')