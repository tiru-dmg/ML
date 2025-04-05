"""Experiment-4:
Exercises to solve the real-world problems using the following
machine learning methods:

 a) Linear Regression
"""



import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X, y = np.array([[500],[1000],[1500],[2000],[2500]]), np.array([100,150,200,250,300])
model = LinearRegression().fit(X, y)
preds = model.predict(X)

print("Mean Squared Error:", round(mean_squared_error(y, preds), 2))
plt.scatter(X, y, c='b', label='Actual')
plt.plot(X, preds, 'r', label='Predicted')
plt.title('Area vs Price'), plt.xlabel('Area'), plt.ylabel('Price'), plt.legend(), plt.grid(), plt.show()

for val in [1200, -300, 'abc', 0]:
    try:
        a = float(val)
        if a <= 0: raise ValueError("Area must be positive.")
        print(f"Predicted price for {a} sq ft: ${model.predict([[a]])[0]:.2f}k")
    except Exception as e:
        print(f"Invalid input '{val}': {e}")
