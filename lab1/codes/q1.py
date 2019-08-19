import numpy as np
import pandas as pd

df = pd.read_csv("/home/sunil/Desktop/sem3/data_science_3/lab1/files/winequality-red.csv")
cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
"""
for col in cols:
    mean = np.mean(data[col])
    print(mean)
"""

print(df['pH'])