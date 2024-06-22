import numpy
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

digits = datasets.load_digits()
images = digits.images

# print(images[7])
_,axes = plt.plot(1, 8, images)
for i in range(8):
    axes[i]

