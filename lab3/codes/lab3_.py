from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd


def read_path(path_to_file):
    return pd.read_csv(path_to_file)
    

def show_boxplot(dataframe):
    l = []
    for col in dataframe.columns:
        l.append(col)
    for i in range(len(l)-1):
        plt.boxplot(dataframe[l[i]])
        plt.xlabel(l[i])
        plt.show()

def replace_outliers(dataframe):
    q1 = dataframe.quantile(0.25)
    q3 = dataframe.quantile(0.75)
    ir = q3 - q1
    l = []      
    for col in dataframe.columns:
        l.append(col)
    mdn = []
    for j in range(len(l)-1):
        mdn.append(dataframe[l[j]].median())
    for i in range(len(l)-1):
        for j in range(len(dataframe)):
            if dataframe.at[j,l[i]] <= (q1[i]-(1.5*ir[i])) or dataframe[l[i]][j] >= (q3[i]+(1.5*ir[i])):
                dataframe.at[j,l[i]] = mdn[i]
    return dataframe            
                
def range_df(attribute_name, dataframe):
    return list([min(dataframe[attribute_name]),max(dataframe[attribute_name])])

def minmaxnorm(dataframe,a,b):
    dataframe2 = dataframe
    l = []      
    for col in dataframe2.columns:
        l.append(col)
    mnx = []
    for i in range(len(l)-1):
        mnx.append([min(dataframe[l[i]]),max(dataframe[l[i]])])
    for i in range(len(l)-1):
        for j in range(len(dataframe2)):
            dataframe2.at[j,l[i]] = (b-a)*(dataframe.at[j,l[i]] - mnx[i][0])/(mnx[i][1] - mnx[i][0]) + a
    return dataframe2

def standardize(dataframe):
    dataframe2 = dataframe
    l = []      
    for col in dataframe2.columns:
        l.append(col)
    dataframe2 = dataframe
    mst3 = []
    for i in range(len(l)-1):
        mst3.append([dataframe[l[i]].mean(),dataframe[l[i]].std()])
    
    for i in range(len(l)-1):
        for j in range(len(dataframe)):
            dataframe2.at[j,l[i]] = (dataframe.at[j,l[i]]-mst3[i][0])/(mst3[i][1])
    return dataframe2

def main():
    df = read_path("/home/sunil/Desktop/sem3/ds3/data_science_3/lab3/files/winequality_red_original.csv")
    df2=df
    df3=df
    df4=df
    l = []      
    for col in df.columns:
        l.append(col)
    #show_boxplot(df)
    df2 = replace_outliers(df2) 
    print("after replace outliers")
    #show_boxplot(df2)
    print("UUUUUUUUUUUUUUUUUUUUUUUUUUU")
    print(df.head())
    df3 = minmaxnorm(df2,0,1)
    print("fffffffffffffffffffffffffffff")
    print(df.head())
    df4 = minmaxnorm(df,0,20)
    df5 = standardize(df)
    lst = range_df(l[0], df)
    print(df2.head())
    print(df3.head())
    print(df4.head())
    print(df5.head())
    print(lst)

if __name__=="__main__":
    main()
    
