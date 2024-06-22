import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({'X':[1, 2, None, 3], 'y':[2, 2, 4, 6]})
print(df)
df_b = df.fillna(0)
print(df_b)
df_m = df.fillna(df.mean())
print(df_m)
linear_regression = LinearRegression().fit(df_b[['X']], df_b[['y']])

predictions = linear_regression.predict([[1], [2], [5]])
print(predictions)
