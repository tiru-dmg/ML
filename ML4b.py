"""Experiment-4:
Exercises to solve the real-world problems using the following
machine learning methods:

 a) Logistic Regression
"""




from sklearn.linear_model import LogisticRegression
import numpy as np, matplotlib.pyplot as plt

X = np.array([[22,30000],[25,32000],[47,80000],[52,110000],[46,95000],[56,130000],[23,40000],[48,70000]])
y = np.array([0,0,1,1,1,1,0,1])

model = LogisticRegression().fit(X, y)

print("✅ Valid Input:", [[28,50000]], "Prediction (Buy=1):", model.predict([[28,50000]])[0])

try: model.predict([[28]])
except Exception as e: print("❌ Invalid Input Error:", e)

xx, yy = np.meshgrid(np.arange(X[:,0].min()-5, X[:,0].max()+5, 1),
                     np.arange(X[:,1].min()-5000, X[:,1].max()+5000, 1000))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('Age'); plt.ylabel('Salary'); plt.title('Logistic Regression Decision Boundary')
plt.show()
