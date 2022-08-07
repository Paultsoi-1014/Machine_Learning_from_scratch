import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from algorithm import LinearRegression_GD, LinearRegression_NE

# Model
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
plt.scatter(X_train, y_train,s=30, marker='x')
plt.show()
'''

LR_GD = LinearRegression_GD()
LR_GD.fit(X_train, y_train)
y_pred_GD = LR_GD.predict(X_test)

LR_NE = LinearRegression_NE()
LR_NE.fit(X_train, y_train)
y_pred_NE = LR_NE.predict(X_test)

plt.plot(X_test, y_pred_GD, c='b')
plt.plot(X_test, y_pred_NE, c='r')
plt.scatter(X_test, y_test, s=20, marker='x')
plt.show()

'''
# Model
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
'''

