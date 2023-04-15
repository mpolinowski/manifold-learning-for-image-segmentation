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

## Local Linear Embedding

```python
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
```

### Digits Dataset

```python
# load digits dataset with labels
X,y = load_digits(return_X_y=True)
X.shape
# (images, features)
# (1797, 64)
```

```python
plt.figure(figsize=(8,8))
plt.title('Image Label: ' + str(y[888]))
plt.imshow(X[888].reshape(8,8))
```

![Dimensionality Reduction for Image Segmentation](https://github.com/mpolinowski/manifold-learning-for-image-segmentation/blob/master/assets/Dimensionality_Reduction_for_Image_Segmentation_01.png)

```python
fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12,12))
axes[0,0].title.set_text('Image Label: ' + str(y[111]))
axes[0,0].imshow(X[111].reshape(8,8), cmap='Greens')
axes[0,1].title.set_text('Image Label: ' + str(y[222]))
axes[0,1].imshow(X[222].reshape(8,8), cmap='Blues')
axes[0,2].title.set_text('Image Label: ' + str(y[333]))
axes[0,2].imshow(X[333].reshape(8,8), cmap='Reds')
axes[1,0].title.set_text('Image Label: ' + str(y[444]))
axes[1,0].imshow(X[444].reshape(8,8), cmap='Blues')
axes[1,1].title.set_text('Image Label: ' + str(y[555]))
axes[1,1].imshow(X[555].reshape(8,8))
axes[1,2].title.set_text('Image Label: ' + str(y[666]))
axes[1,2].imshow(X[666].reshape(8,8), cmap='Blues')
axes[2,0].title.set_text('Image Label: ' + str(y[777]))
axes[2,0].imshow(X[777].reshape(8,8), cmap='Reds')
axes[2,1].title.set_text('Image Label: ' + str(y[888]))
axes[2,1].imshow(X[888].reshape(8,8), cmap='Blues')
axes[2,2].title.set_text('Image Label: ' + str(y[999]))
axes[2,2].imshow(X[999].reshape(8,8), cmap='Greens')
plt.tight_layout()
```

![Dimensionality Reduction for Image Segmentation](https://github.com/mpolinowski/manifold-learning-for-image-segmentation/blob/master/assets/Dimensionality_Reduction_for_Image_Segmentation_02.png)


### 2-Dimensional Plot

```python
# the dataset has 1797 images with 64 dimensions
# we use LLE to reduce the dimensionality of the dataset
# to help us visualize / classify it
no_components=2
no_neighbors=10

lle = LocallyLinearEmbedding(n_components=no_components, n_neighbors=no_neighbors)
X_lle = lle.fit_transform(X, y=y)

data = pd.DataFrame({'LLE1': X_lle[ :,0], 'LLE2': X_lle[ :,1], 'Class': y})

plt.figure(figsize=(12, 10))
plt.title('2d Plot with 10 nearest neighbors')
sns.scatterplot(x='LLE1', y='LLE2', hue='Class', data=data, palette='tab10')
```

![Dimensionality Reduction for Image Segmentation](https://github.com/mpolinowski/manifold-learning-for-image-segmentation/blob/master/assets/Dimensionality_Reduction_for_Image_Segmentation_03.png)

```python
no_neighbors=15

lle = LocallyLinearEmbedding(n_components=no_components, n_neighbors=no_neighbors)
X_lle = lle.fit_transform(X, y=y)

data = pd.DataFrame({'LLE1': X_lle[ :,0], 'LLE2': X_lle[ :,1], 'Class': y})

plt.figure(figsize=(12, 10))
plt.title('2d Plot with 15 nearest neighbors')
sns.scatterplot(x='LLE1', y='LLE2', hue='Class', data=data, palette='tab10')
```

![Dimensionality Reduction for Image Segmentation](https://github.com/mpolinowski/manifold-learning-for-image-segmentation/blob/master/assets/Dimensionality_Reduction_for_Image_Segmentation_04.png)

```python
no_neighbors=20

lle = LocallyLinearEmbedding(n_components=no_components, n_neighbors=no_neighbors)
X_lle = lle.fit_transform(X, y=y)

data = pd.DataFrame({'LLE1': X_lle[ :,0], 'LLE2': X_lle[ :,1], 'Class': y})

plt.figure(figsize=(12, 10))
plt.title('2d Plot with 20 nearest neighbors')
sns.scatterplot(x='LLE1', y='LLE2', hue='Class', data=data, palette='tab10')
```

![Dimensionality Reduction for Image Segmentation](https://github.com/mpolinowski/manifold-learning-for-image-segmentation/blob/master/assets/Dimensionality_Reduction_for_Image_Segmentation_05.png)


### 3-Dimensional Plot

```python
no_components=3
no_neighbors=10

lle = LocallyLinearEmbedding(n_components=no_components, n_neighbors=no_neighbors)
X_lle = lle.fit_transform(X, y=y)

data = pd.DataFrame({
    'LLE1': X_lle[ :,0],
    'LLE2': X_lle[ :,1],
    'LLE3': X_lle[ :,2],
    'Class': y})

# data.head()

plot = px.scatter_3d(
    data,
    x = 'LLE1',
    y = 'LLE2',
    z = 'LLE3',
    color='Class')

plot.show()
```

![Dimensionality Reduction for Image Segmentation](https://github.com/mpolinowski/manifold-learning-for-image-segmentation/blob/master/assets/Dimensionality_Reduction_for_Image_Segmentation_06.png)

```python
no_components=3
no_neighbors=15

lle = LocallyLinearEmbedding(n_components=no_components, n_neighbors=no_neighbors)
X_lle = lle.fit_transform(X, y=y)

data = pd.DataFrame({
    'LLE1': X_lle[ :,0],
    'LLE2': X_lle[ :,1],
    'LLE3': X_lle[ :,2],
    'Class': y})

# data.head()

plot = px.scatter_3d(
    data,
    x = 'LLE1',
    y = 'LLE2',
    z = 'LLE3',
    color='Class')

plot.show()
```

![Dimensionality Reduction for Image Segmentation](https://github.com/mpolinowski/manifold-learning-for-image-segmentation/blob/master/assets/Dimensionality_Reduction_for_Image_Segmentation_07.png)

```python
no_components=3
no_neighbors=20

lle = LocallyLinearEmbedding(n_components=no_components, n_neighbors=no_neighbors)
X_lle = lle.fit_transform(X, y=y)

data = pd.DataFrame({
    'LLE1': X_lle[ :,0],
    'LLE2': X_lle[ :,1],
    'LLE3': X_lle[ :,2],
    'Class': y})

# data.head()

plot = px.scatter_3d(
    data,
    x = 'LLE1',
    y = 'LLE2',
    z = 'LLE3',
    color='Class')

plot.show()
```

![Dimensionality Reduction for Image Segmentation](https://github.com/mpolinowski/manifold-learning-for-image-segmentation/blob/master/assets/Dimensionality_Reduction_for_Image_Segmentation_08.png)

```python

```
