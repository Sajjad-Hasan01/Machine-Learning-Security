import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import pandas as pd

# Load the iris data
iris = datasets.load_iris()
# print(iris)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df)
df['target'] = iris.target
x = iris.data[:, :2]
y = (iris.target == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
Perceptron_model1 = Perceptron(max_iter=100, tol=1e-3)
Perceptron_model1.fit(X_train, y_train)
y_pred = Perceptron_model1.predict(X_test)
plt.scatter(iris.data[:,0], iris.data[:,1], c=y, cmap="Paired")
plt.show()
score = Perceptron_model1.score(X_test, y_test)
print("Accuracy:", score)
