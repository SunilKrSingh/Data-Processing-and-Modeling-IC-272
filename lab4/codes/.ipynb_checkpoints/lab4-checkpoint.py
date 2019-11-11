import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np


def load_dataset(path_to_file):
    return pd.read_csv(path_to_file)

def normalize(df):
    scaler = MinMaxScaler()
    arrNorm = scaler.fit_transform(df)
    dfNorm = pd.DataFrame(arrNorm)
    return dfNorm

def standardize(df):
    scaler = StandardScaler()
    Std = scaler.fit_transform(df)
    dfStd = pd.DataFrame(Std)
    return dfStd

def shuffle_df(df):
    return shuffle(df)

def test_train_split(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:,-1].values
    #X = np.array(df.iloc[:, 0:-1])
    #y = np.array(df['Z_Scratch'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    np.savetxt("/home/sunil/Desktop/sem3/ds3/data_science_3/lab5/X_train.csv",X_train,delimiter=',')
    np.savetxt("/home/sunil/Desktop/sem3/ds3/data_science_3/lab5/X_test.csv",X_test,delimiter=',')
    np.savetxt("/home/sunil/Desktop/sem3/ds3/data_science_3/lab5/y_train.csv",y_train,delimiter=',')
    np.savetxt("/home/sunil/Desktop/sem3/ds3/data_science_3/lab5/y_test.csv",y_test,delimiter=',')
    print(type(X_train))
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracy = (accuracy_score(y_test, pred))
    print(accuracy)
    
    print(confusion_matrix(y_test, pred))
def classification(k):
    X_train = np.array(load_dataset("/home/sunil/Desktop/sem3/ds3/data_science_3/lab5/X_train.csv"))
    X_test = np.array(load_dataset("/home/sunil/Desktop/sem3/ds3/data_science_3/lab5/X_test.csv"))
    y_train = np.array(load_dataset("/home/sunil/Desktop/sem3/ds3/data_science_3/lab5/y_train.csv"))
    y_test = np.array(load_dataset("/home/sunil/Desktop/sem3/ds3/data_science_3/lab5/y_test.csv"))
    
    accuracies = []
    for i in range(len(k)):
        knn = KNeighborsClassifier(n_neighbors=k[i])
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        accuracy = (accuracy_score(y_test, pred))
        accuracies.append(accuracy)
        print("k = ",k[i]," : ")
        print(confusion_matrix(y_test, pred))
    return accuracies

def main():
    path_to_file = "/home/sunil/Desktop/sem3/ds3/data_science_3/lab5/inLab/SteelPlateFaults-2class.csv"
    df = load_dataset(path_to_file)
    
    dfN = df.copy()
    dfS = df.copy()
    
    k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
    
    test_train_split(df)
    dfAcc = classification(k)
    
    dfNorm = normalize(dfN)
    test_train_split(dfNorm)
    #dfNormAcc = classification(k)
    
    dfStd = standardize(dfS)
    test_train_split(dfStd)
    #dfStdAcc = classification(k)
    
    print()
    print("accuracy matrix for Original Data: \n",dfAcc)
    print()
    #print("accuracy matrix for Normalized Data: \n",dfNormAcc)
    print()
    #print("accuracy matrix for Standardized Data: \n",dfStdAcc)
    
    plt.plot(k,dfAcc,label = "org")
    #plt.plot(k,dfNormAcc,label = "norm")
    #plt.plot(k,dfStdAcc,label = "standard")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()