import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
df = pd.DataFrame(digits.data)
df['digits'] = digits.target
# print(df)
# print(df['digits'])

images = digits.images
# print(images[1])
# print(images[2])

_, axes = plt.subplots(1, 4)
images_abd_labels = list((digits.images, digits.target))
for i in range(4):
    numbers = digits.target
    axes[i].set_title('Image: %i' % numbers[i])
    axes[i].imshow(images[i], cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()
