import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

correct = 0
wrong = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        correct += 1
        print(f"Correct prediction: Actual class {y_test[i]}, Predicted class {y_pred[i]}")
    else:
        wrong += 1
        print(f"Wrong prediction: Actual class {y_test[i]}, Predicted class {y_pred[i]}")

print(f"\nNumber of correct predictions: {correct}")
print(f"Number of wrong predictions: {wrong}")
