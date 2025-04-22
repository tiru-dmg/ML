'''Write a program to Implement Principle Component Analysis'''

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
df = pd.DataFrame(PCA(n_components=2).fit_transform(X), columns=['PC1', 'PC2'])
df['target'] = y

plt.figure(figsize=(8,6))
for t, c in zip([0,1,2], ['r','g','b']):
    plt.scatter(df[df['target']==t]['PC1'], df[df['target']==t]['PC2'], c=c, s=50)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend([0,1,2])
plt.show()
