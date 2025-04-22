/*Experiment-14:
Write a program to Implement Support Vector Machines and Principle
Component Analysis*/

import numpy as np, pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(X_train_pca, y_train)
y_pred = svm_clf.predict(X_test_pca)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))

def plot_svm_decision_boundary(X, y, model):
    h = .02
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolors='k')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('SVM Decision Boundary with PCA')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    plt.show()

plot_svm_decision_boundary(X_test_pca, y_test, svm_clf)
