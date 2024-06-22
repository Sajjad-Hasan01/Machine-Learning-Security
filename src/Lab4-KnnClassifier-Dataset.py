import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = datasets.load_iris()
x = df.data
y = df.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
colormap = ListedColormap(['r','g','b'])
plt.figure()
plt.scatter(x[:,0],x[:,1],c=y,cmap=colormap)
plt.show()
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
print(pred)
acc = np.sum(y_test == pred) / len(y_test)

print('Accuracy: ', acc)
print('Accuracy: ', accuracy_score(pred, y_test))
