
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import statsmodels.api as sm 
from sklearn.metrics import mean_squared_error as mse
from math import sqrt 
from statsmodels.tsa.ar_model import AR 

#Q1 I guess
f = pd.read_csv("Rain.csv")
cols = ["Date","Rain(mm)"]
d = f["Date"].values
t = f["Rain(mm)"].values
tp1 = t[1:len(t)]   #present day temperature
tp2 = t[0:len(t)-1]   #previous day temperature
print("Correlation between previous day and today: ",np.corrcoef(tp1,tp2)[0,1])

f.plot()
plt.show()
#Q2 I guess
sm.graphics.tsa.plot_acf(t, lags = 30)
plt.show()

#Q3 I guess
ftr = f[0:int(len(f)/2)]
ft = f[int(len(f)/2):]

ft1 = ft[1:]
ft2 = ft[0:-1]

print("Persistance model RMSE test: ",sqrt(mse(ft1["Rain(mm)"].values,ft2["Rain(mm)"].values)))

ft1 = ftr[1:]
ft2 = ftr[0:-1]

print("Persistance model RMSE train: ",sqrt(mse(ft1["Rain(mm)"].values,ft2["Rain(mm)"].values)))

#Q4 I guess
train, test = ftr,ft
train = train["Rain(mm)"].values
test = test["Rain(mm)"].values

model = AR(train)
model_fit = model.fit()
print('Lag: ', model_fit.k_ar)
print('Coefficients: ', model_fit.params)

predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)

error = sqrt(mse(test, predictions))
print('Test RMSE',error)
