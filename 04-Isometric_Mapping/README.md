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

## Isometric Mapping

```python
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
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
k_nearest_neighbors = 10

isomap = Isomap(
    n_components=no_components,
    n_neighbors=k_nearest_neighbors)

X_iso = isomap.fit_transform(X)
print('Reconstruction Error: ', isomap.reconstruction_error())
# Reconstruction Error:  3092.669294495556

data = pd.DataFrame({
    'ISOMAP1': X_iso[:,0],
    'ISOMAP2': X_iso[:,1],
    'Class': y})
```

```python
fig = plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='ISOMAP1',
    y='ISOMAP2',
    hue='Class',
    data=data,
    palette='tab10')
```

![Isometric Mapping](../assets/Dimensionality_Reduction_for_Image_Segmentation_13.png)


### 3-Dimensional Plot

```python
no_components = 3
k_nearest_neighbors = 10

isomap = Isomap(
    n_components=no_components,
    n_neighbors=k_nearest_neighbors)

X_iso = isomap.fit_transform(X)
print('Reconstruction Error: ', isomap.reconstruction_error())
# Reconstruction Error:  2522.73434274533

data = pd.DataFrame({
    'ISOMAP1': X_iso[:,0],
    'ISOMAP2': X_iso[:,1],
    'ISOMAP3': X_iso[:,2],
    'Class': y})
```

```python
plot = px.scatter_3d(
    data,
    x = 'ISOMAP1',
    y = 'ISOMAP2',
    z = 'ISOMAP3',
    color='Class')

plot.show()
```

![Isometric Mapping](../assets/Dimensionality_Reduction_for_Image_Segmentation_14.png)
