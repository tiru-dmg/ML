"""Experiment-4:
Exercises to solve the real-world problems using the following
machine learning methods:

 a) BINARY CLASSIFIERS
"""




from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = load_iris(return_X_y=True)
X, y = X[y != 2], y[y != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for name, clf in [("ONE VS REST", OneVsRestClassifier), ("ONE VS ONE", OneVsOneClassifier)]:
    model = clf(LogisticRegression()).fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n{name}\nAccuracy: {accuracy_score(y_test, pred)}\nSample Predictions: {pred[:5]}\nConfusion Matrix:\n{confusion_matrix(y_test, pred)}")
