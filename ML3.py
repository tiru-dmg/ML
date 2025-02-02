import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report

iris = load_iris()
X, y, names = iris.data, iris.target, iris.target_names
print(pd.DataFrame(np.c_[X, y], columns=iris.feature_names + ['species']).head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42).fit(X_train, y_train)
print("Test Accuracy:", clf.score(X_test, y_test))
print(classification_report(y_test, clf.predict(X_test), target_names=names))
print("Predicted Species:", names[clf.predict([[5.1, 3.5, 1.4, 0.2]])][0])

plt.figure(figsize=(6, 4))
plot_tree(clf, feature_names=iris.feature_names, class_names=names, filled=True, max_depth=2)
plt.show()
