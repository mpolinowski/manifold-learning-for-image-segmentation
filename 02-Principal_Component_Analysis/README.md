---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Dimensionality Reduction for Image Segmentation

Use [Manifold Learning](https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html) and the [LD Analysis](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2023-04-13-fisher-discriminant-analysis/2023-04-13) to Visualize Image Datasets.

##  Principal Component Analysis

```python
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
```

### Digits Dataset

```python
# load digits dataset with labels
X,y = load_digits(return_X_y=True)
X.shape
# (images, features)
# (1797, 64)
```

### 2-Dimensional Plot

```python
no_components = 2
pca = PCA(n_components=no_components).fit(X)
X_pca = pca.transform(X)

data = pd.DataFrame({
    'PCA1': X_pca[:,0],
    'PCA2': X_pca[:,1],
    'Class': y})

fig = plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Class',
    palette='tab10',
    data=data)
```

![Principal Component Analysis](https://github.com/mpolinowski/manifold-learning-for-image-segmentation/blob/master/assets/Dimensionality_Reduction_for_Image_Segmentation_09.png)


### 3-Dimensional Plot

```python
no_components = 3
pca = PCA(n_components=no_components).fit(X)
X_pca = pca.transform(X)

data = pd.DataFrame({
    'PCA1': X_pca[:,0],
    'PCA2': X_pca[:,1],
    'PCA3': X_pca[:,2],
    'Class': y})

plot = px.scatter_3d(
    data,
    x = 'PCA1',
    y = 'PCA2',
    z = 'PCA3',
    color='Class')

plot.show()
```

![Principal Component Analysis](https://github.com/mpolinowski/manifold-learning-for-image-segmentation/blob/master/assets/Dimensionality_Reduction_for_Image_Segmentation_10.png)

```python

```
