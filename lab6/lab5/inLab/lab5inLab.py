import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_dataset(path_to_file):
     df=pd.read_csv(path_to_file)
     return df

def normalization(df):
     scaler=MinMaxScaler(feature_range=(0,1),copy=True)
     df_norm=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
     df_norm[df.columns[-1]]=df[df.columns[-1]]
     return df_norm

def standardize(df):
     scaler=StandardScaler()
     df_std=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
     df_std[df.columns[-1]]=df[df.columns[-1]]
     return df_std

def train_test(df):
     x=df.drop(df.columns[-1],axis=1)
     y=df[df.columns[-1]]
     xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 42)
     return([xTrain,xTest,yTrain,yTest])

def classification(k,xTrain,yTrain,xTest):
         knn = KNeighborsClassifier(n_neighbors=k)
         knn.fit(xTrain, yTrain)
         yPred = knn.predict(xTest)
         return yPred

def percentage_accuracy(yPred,yTest):
     return(accuracy_score(yTest, yPred))


def confusion_matrixp(yPred,yTest):
     return(confusion_matrix(yTest,yPred))


df=load_dataset("../files/pima-indians-diabetes.csv")
df_norm=normalization(df)
df_std=standardize(df)

def main(df,l,c):
     #print(l)
     xTrain=train_test(df)[0]
     xTest=train_test(df)[1]
     yTrain=train_test(df)[2]
     yTest=train_test(df)[3]

     s=[]
     for i in range(1,22,2):
         print(i,":")
         yPred=classification(i,xTrain,yTrain,xTest)
         print(confusion_matrixp(yPred,yTest))
         print(percentage_accuracy(yPred,yTest))
         s.append(percentage_accuracy(yPred,yTest))
         print("\n")
     print(s.index(np.max(s))*2+1)
     print("\n")
     plt.plot(range(1,22,2),s,label=l,color=c)
     print(df.columns[-1])

main(df,"original","red")
main(df_norm,"normalization","blue")
main(df_std,"standarize","green")
plt.legend()
plt.show()