#lab 5
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np


def load_dataset(path_to_file):
    return pd.read_csv(path_to_file)

def outliers_detection(df,l):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    cnt = 0
    l = [cols for cols in df.columns()]
    for i in range(len(l)):
        for j in range(len(df)):
            if df.at[j,l[i]] <= (q1[i]-(1.5*iqr[i])) or df[l[i]][j] >= (q3[i]+(1.5*iqr[i])):
                cnt += 1
    return cnt

def normalize(dff,l):
    df = dff
    mm = MinMaxScaler()
    for i in range(len(l)-1):
        df[l[i]] = pd.DataFrame(mm.fit_transform(pd.DataFrame(dff[l[i]])),columns=[l[i]])
    df.to_csv("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/normalize_of_df.csv")
    return df

def standardize(dff,l):
    df = dff
    ss = StandardScaler()
    for i in range(len(l)-1):
        df[l[i]] = pd.DataFrame(ss.fit_transform(pd.DataFrame(dff[l[i]])),columns=[l[i]])
    df.to_csv("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/standardize_of_df.csv")
    return df

def shuffle_df(df):
    return shuffle(df)
    

def test_train_split_3(df):
    X1 = df.iloc[:, :-1].values
    X_LABEL1 = df.iloc[:,-1].values
    X_train1, X_test1, X_LABEL_train1, X_LABEL_test1 = train_test_split(X1, X_LABEL1, test_size=0.3, random_state=42)
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/diabetes-train-normalise.csv",X_train1,delimiter=',')
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/diabetes-test-normalise.csv",X_test1,delimiter=',')
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-train-normalise.csv",X_LABEL_train1,delimiter=',')
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-test-normalise.csv",X_LABEL_test1,delimiter=',')
    
def classification_3(k):
    X_train = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/diabetes-train-normalise.csv")
    X_test = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/diabetes-test-normalise.csv")
    X_LABEL_train = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-train-normalise.csv")
    X_LABEL_test = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-test-normalise.csv")
    acc = []
    print("#############################")
    print("norm")
    for i in range(len(k)):
        knn = KNeighborsClassifier(n_neighbors=k[i])
        knn.fit(X_train, X_LABEL_train)
        y_pred = knn.predict(X_test)
        acc.append(metrics.accuracy_score(X_LABEL_test, y_pred))
        print("k = ",k[i]," : ")
        print(confusion_matrix(X_LABEL_test, y_pred))
    return acc

def test_train_split_4(df):
    X = df.iloc[:, :-1].values
    X_LABEL = df.iloc[:,-1].values
    X_train, X_test, X_LABEL_train, X_LABEL_test = train_test_split(X, X_LABEL, test_size=0.3, random_state=42)
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_diabetes-train-standardise.csv",X_train,delimiter=',')
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_diabetes-test-standardise.csv",X_test,delimiter=',')
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-train-standardise.csv",X_LABEL_train,delimiter=',')
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-test-standardise.csv",X_LABEL_test,delimiter=',')
    
def classification_4(k):
    X_train = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_diabetes-train-standardise.csv")
    X_test = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_diabetes-test-standardise.csv")
    X_LABEL_train = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-train-standardise.csv")
    X_LABEL_test = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-test-standardise.csv")
    acc = []
    print("#############################")
    print("standard")
    for i in range(len(k)):
        knn = KNeighborsClassifier(n_neighbors=k[i])
        knn.fit(X_train, X_LABEL_train)
        y_pred = knn.predict(X_test)
        acc.append(metrics.accuracy_score(X_LABEL_test, y_pred))
        print("k = ",k[i]," : ")
        print(confusion_matrix(X_LABEL_test, y_pred))
    return acc       

def test_train_split_2(df):
    X = df.iloc[:, :-1].values
    X_LABEL = df.iloc[:,-1].values
    X_train, X_test, X_LABEL_train, X_LABEL_test = train_test_split(X, X_LABEL, test_size=0.3, random_state=42)
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/diabetes-train.csv",X_train,delimiter=',')
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/diabetes-test.csv",X_test,delimiter=',')
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-train.csv",X_LABEL_train,delimiter=',')
    np.savetxt("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-test.csv",X_LABEL_test,delimiter=',')
    
def classification_2(k):
    X_train = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/diabetes-train.csv")
    X_test = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/diabetes-test.csv")
    X_LABEL_train = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-train.csv")
    X_LABEL_test = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/X_LABEL_diabetes-test.csv")
    acc = []
    print("#############################")
    print("original")
    for i in range(len(k)):
        knn = KNeighborsClassifier(n_neighbors=k[i])
        knn.fit(X_train, X_LABEL_train)
        y_pred = knn.predict(X_test)
        acc.append(metrics.accuracy_score(X_LABEL_test, y_pred))
        print("k = ",k[i]," : ")
        print(confusion_matrix(X_LABEL_test, y_pred))
    return acc



def main():
    df1 = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/pima-indians-diabetes.csv")
    #dff1 = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/pima-indians-diabetes.csv")
    #dff2 = load_dataset("/home/yashwinder/Documents/SEM_3/Data_Science_3_(IC2XX)/lab/lab_5/pima-indians-diabetes.csv")

    dff1 = df1.copy()
    dff2 = df1.copy()
    
    l = []
    k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
    for col in df1.columns:
        l.append(col)
    df3 = normalize(dff1,l)
    
    df4 = standardize(dff2,l)
    test_train_split_2(df1)
    a = classification_2(k)
    
    test_train_split_3(df3)
    b = classification_3(k)
    
    test_train_split_4(df4)
    c = classification_4(k)   
    print("###################")
    print("accuracy matrix for original",a)
    print("###################")
    print("accuracy matrix for norm",b)
    print("###################")
    print("accuracy matrix for standard",c)
    
    plt.plot(k,a,label = "org")
    plt.plot(k,b,label = "norm")
    plt.plot(k,c,label = "standard")
    plt.legend()
    plt.show()
   

