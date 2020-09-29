##kodingan ini tentang prinsipal data analisis, dataset yang dipakai adalah dataset iris yang bisa didapat dari library sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sb
from IPython.display import Image
from IPython.core.display import HTML
from pylab import rcParams
import sklearn
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import datasets

rcParams['figure.figsize']=5,4
sb.set_style('whitegrid')


##mengambil data dari dataset iris
iris = datasets.load_iris()
X = iris.data
variable_names = iris.feature_names


##menggunakan PCA(prinsipal component analisis)
pca = decomposition.PCA()
iris_pca = pca.fit_transform(X)
pca.explained_variance_ratio_

##menampilkan heatmap
comps = pd.DataFrame(pca.components_,columns=variable_names)
sb.heatmap(comps)
