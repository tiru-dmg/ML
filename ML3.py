"""Experiment-3:
Write a program to demonstrate the working of the decision tree
based ID3 algorithm. Use an appropriate data set for building the
decision tree and apply this knowledge to classify a new sample.
"""


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [2, 1]
]
y = [0, 0, 1, 1]

print("First few rows of dataset:")
for i in range(len(X)):
    print(f"Weather: {X[i][0]}, Temperature: {X[i][1]} => PlayTennis: {'Yes' if y[i] else 'No'}")

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print("\nAccuracy of model on training data:", accuracy)
print("\nClassification Report:\n", classification_report(y, y_pred, target_names=["No", "Yes"]))

sample = [[1, 0]]
predicted_class = model.predict(sample)[0]
print(f"\nPrediction for sample (Overcast, Hot): {'Yes' if predicted_class else 'No'}")

plt.figure(figsize=(8, 5))
plot_tree(model, feature_names=["Weather", "Temperature"], class_names=["No", "Yes"], filled=True)
plt.title("Decision Tree for PlayTennis")
plt.show()
