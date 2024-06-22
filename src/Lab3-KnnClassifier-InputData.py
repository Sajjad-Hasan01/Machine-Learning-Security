import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#   Calc distance between two points
def knn_2_points():
    print('Enter point A:')
    x1, y1 = map(int, input().split())

    print('Enter point B:')
    x2, y2 = map(int, input().split())

    dist = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    print(f'Distance = {str(dist)}')

def knn_model():
    X_train = np.array([[2,2], [2,4], [4,5], [5,6], [10,6], [8,8]])
    y_train = np.array([1, 0, 1, 0, 0, 0])

    X_test = np.array([[8,7], [2,5], [4,9], [6,11], [10,6]])

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)   # Training
    y_test = knn.predict(X_test)    # Testing

    # for test, pred in X_test, y_test:
    #     print(f'Class of {test} is {pred}')
    #     print(f'Class of {X_test[i]} is {y_test[i]}')

    print(f'Class of \n{X_test}\n{y_test}')


knn_model()
# K = 3 -> [0 1 0 0 0]
# K = 5 -> [0 0 0 0 0]
