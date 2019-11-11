

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:04:13 2019

@author: rupanshirupanshi
"""
import statistics as st
import pandas as pd
import numpy as np;
import scipy.stats as ss
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import mixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import  datasets
from sklearn import decomposition as dem
from sklearn.cluster import DBSCAN 
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import math as m
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
import scipy.stats as st
from sklearn.metrics import mean_squared_error
import seaborn as ss
from sklearn.decomposition import PCA
#from feature_selector import FeatureSelector

def read_data(path):
    return(pd.read_csv(path))

def info(df):
    cols=df.columns
    print("mean = ",df.mean(),'\n')
    print("median = ",df.median(),'\n')
    print("standard deviation = ",df.std(),'\n')
    for i in cols:
        print("mode = ",st.mode(df[i].values,axis=0))

def prediction_accuracy(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# first lets start with preprocessing of data
def check_null_val(data):
    print(data.isnull().sum())
    ##there is no null values present
###############################################################
def outlier_detection(df):
    y = list(df.columns)
    #y.remove("CreationTime") ##it is a categorical data
    z = 0
    for i in y:
        s=df[i]
        q1=df[i].quantile(0.25)
        q3=df[i].quantile(0.75)
        iqr=q3-q1
        c=0
        for j in range(len(df[i])):
            if (q1-1.5*iqr)>=s[j] or s[j]>=(q3+1.5*iqr):
                    c+=1
        print("no. of outliers in ",i," is ",c)
        z = z+c
    print("total no. of outliers ",z)
def boxplt(data):
    data.boxplot()
###############################################################
def standardization(data):
    df1 = data.copy()
    x = list(df1.columns)
    #x.remove("CreationTime")
    for i in x:
        df1[i] = ((df1[i] - df1[i].mean())/df1[i].std())
    return df1
###############################################################
def normalise(data,newmax,newmin):
    df1 = data.copy()
    x = list(data.columns)
    #x.remove("CreationTime")
    for i in x:
        df1[i] = (((df1[i] - df1[i].min())/(df1[i].max() - df1[i].min()))*(newmax-newmin))+newmin
    return df1
###############################################################
def replace_outlier(data):
    df1 = data.copy()
    y = list(df1.columns)
    #y.remove("CreationTime")     ##it is a categorical data
    for i in y:
        s=df1[i]
        q1=df1[i].quantile(0.25)
        q3=df1[i].quantile(0.75)
        iqr=q3-q1
        c=df1[i].median()
        for j in range(len(df1[i])):
            if (q1-1.5*iqr)>=s[j] or s[j]>=(q3+1.5*iqr):
                df1.loc[j,i]=c
    return df1
##############################################################
def corr_analysis(data):
    col = list(data.columns)
    #col.remove("CreationTime")
    col.remove( "InBandwidth")
    target_attribute = "InBandwidth"
    CORRs = []
    print("\nCorrelation Coefficient of 'InBandwidth' with other attributes:\n")
    for i in col:
        cor = abs(st.pearsonr(data[target_attribute],data[i])[0])
        CORRs.append(cor)
        print(i,": ",cor) 
    return CORRs
###############################################################
def PcA(data,i):
    df1 = data.copy()
    #df1.drop(columns = ["CreationTime"], inplace = True)
    df1.drop(columns = ["InBandwidth"], inplace = True)
    pca = PCA(n_components=i)
    newdata = pd.DataFrame(pca.fit_transform(df1))
    newdata2=pd.concat([newdata,data[['InBandwidth']]], axis = 1)
    return newdata2
###############################################################
def feature_selection(data):
    col = list(data.columns)
    #col.remove("CreationTime")
    col.remove( "InBandwidth")
    #remove the attributes with more than 50% missing values
    cor=(corr_analysis(data))
    plt.figure(figsize=(12,10))
    corX = data.corr()
    ss.heatmap(corX, annot=True, cmap=plt.cm.Reds)
    plt.show()
    irrelevant_features=[]
    for i in range(len(col)):
        if(cor[i]<0.5):
            irrelevant_features.append(col[i])
    for i in range(len(irrelevant_features)):
        data.drop(irrelevant_features[i],1)
    print(irrelevant_features)
    return data

#####end of data preprocessing
    



### data descriptive analysis
    

###data predictive analysis 
## inbandwidth is dependent attribute
def split_for_reg(df):
    train,test = train_test_split(df,train_size = 0.70,test_size = 0.30,random_state= 42,shuffle = True)
    return train,test

def mul_lin_reg(train,test):
    print("Multiple Linear regression\n")
    ### categorical data removal
    #train.drop(columns = ["CreationTime"], inplace = True)
    #test.drop(columns = ["CreationTime"], inplace = True)
    ###
    Y = train["InBandwidth"]
    train.drop(columns =["InBandwidth"],inplace = True)
    actual = test["InBandwidth"]
    test.drop(columns =["InBandwidth"],inplace = True)
    model = LinearRegression()
    model.fit(train,Y)
    pred = model.predict(test)
    pred_train=model.predict(train)
    print("rmse of test data : ",prediction_accuracy(actual,pred))
    print("rmse of train data : ",prediction_accuracy(Y,pred_train),'\n')
    plt.scatter(actual,pred , color = "blue",  label = 'Test data')
    plt.xlabel('original InBandwidth')
    plt.ylabel('predicted InBandwidth')
    plt.show()
    #############################################################

def poly_reg(train,test):
    ### categorical data removal
    print("Polynomial Curve Fitting\n")
    #train=trainX.drop("CreationTime", axis=1)
    #test=testX.drop("CreationTime", axis=1)
    ###
    Y = train["InBandwidth"]
    train.drop(columns = ["InBandwidth"],inplace =True)
    actual = test["InBandwidth"]
    test.drop(columns = ["InBandwidth"],inplace =True)
    p = [2,3]
    for i in p:
        print("degree of polynomial is ",i)
        poly = PolynomialFeatures(degree = i)
        new_x = poly.fit_transform(train)
        new_test = poly.fit_transform(test)
        reg = LinearRegression()
        reg.fit(new_x,Y)
        pred = reg.predict(new_test)
        print("rmse test : ",sqrt(metrics.mean_squared_error(actual,pred)))
        print("rmse train : ",sqrt(metrics.mean_squared_error(Y,reg.predict(new_x))),'\n')
        plt.scatter(actual,pred , color = "blue",  label = 'Test data')
        plt.xlabel('original InBandwidth')
        plt.ylabel('predicted InBandwidth')
        plt.show()

def corr_column(df):
    col = list(df.columns)
    #col.remove("CreationTime")
    col.remove( "InBandwidth")
    val = []
    for i in col:
        val.append(abs(st.pearsonr(df["InBandwidth"],df[i])[0]))
    x = sorted(val,reverse = True)
    col1 = col[val.index(x[0])]#maximum correalatio coefficient
    col2 = col[val.index(x[1])]
    print(col1,col2)
    return col1,col2

def corr_lin(data,X_train,X_test):
    col1,col2 = corr_column(data)
    X = X_train[[col1,col2]]
    Y = X_train["InBandwidth"]
    test = X_test[[col1,col2]]
    actual = X_test["InBandwidth"]
    reg = LinearRegression()
    reg.fit(X,Y)
    pred_qual = reg.predict(test)
    pred_trn_qual = reg.predict(X)
    pred_trn_err = prediction_accuracy(Y,pred_trn_qual)
    pred_tst_err = prediction_accuracy(actual,pred_qual)
    print("RMSE of training data is ",pred_trn_err)
    print("RMSE of test data is ",pred_tst_err)
    plt.scatter(actual,pred_qual)
    plt.xlabel("test quality")
    plt.ylabel("pred quality")
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111 , projection = '3d')
    ax.scatter(X_train[col1],X_train[col2],Y,color="g",marker=".")
    fig1 = plt.gcf()
    fig1.set_size_inches(10, 10)
    ax.plot_trisurf(X_train[col1],X_train[col2],pred_trn_qual,alpha=0.5)
    ax.scatter(X_train[col1],X_train[col2],pred_trn_qual,color="r",marker="*")
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_zlabel("InBandwidth")
    plt.show()

def main():
    path = "D:\sem3\ds3\data_science_3\project\group8.csv"
    data = pd.read_csv(path)
    data = data.drop("CreationTime", axis=1)
    check_null_val(data)
    outlier_detection(data)
    #boxplt(data)
    df1 = replace_outlier(data)
    
    outlier_detection(df1)
    info(df1)
    #outlier removed dataframe
    stand_df = standardization(df1)  # z-normalised data
    norma_df = normalise(df1,1,0)   # min-max-normalised data
    data_pca=PcA(data,2)
    #outlier_detection(stand_df)           # no. of outliers in each column
    corr_analysis(data)             # corr analysis with target as inbandwidth column
    data_FS=feature_selection(data)
    #########################################################
    #descriptive analysis
    ############################################################
    
    ##data predictive analysis
    train,test = split_for_reg(data)
    print("\nMultiple Linear Regression on untreated data\n")
    mul_lin_reg(train.copy(),test.copy())
    print("Polynomial curve fitting on untreated data")
    poly_reg(train.copy(),test.copy())


    #results with preprocessed datasets when replaced with outliers
    trainP,testP = split_for_reg(df1)
    print("Multiple Linear Regression on data without outliers\n")
    mul_lin_reg(trainP.copy(),testP.copy())
    print("Polynomial curve fitting on data without outliers\n")
    poly_reg(trainP.copy(),testP.copy())

    #resuls on standardized data
    trainS,testS = split_for_reg(stand_df)
    print("Multiple Linear Regression on standardized data\n")
    mul_lin_reg(trainS.copy(),testS.copy())
    print("Polynomial curve fitting on standardized data")
    poly_reg(trainS.copy(),testS.copy())

    #results on normalized data
    trainN,testN = split_for_reg(norma_df)
    print("Multiple Linear Regression on normalized data\n")
    mul_lin_reg(trainN.copy(),testN.copy())
    print("Polynomial curve fitting on normalized data")
    poly_reg(trainN.copy(),testN.copy())

    #results after feature selection
    trainFS,testFS = split_for_reg(data_FS)
    print("Multiple Linear Regression on data after feature selection\n")
    mul_lin_reg(trainFS.copy(),testFS.copy())
    print("Polynomial curve fitting on data after feature selection")
    poly_reg(trainFS.copy(),testFS.copy())

    #results after PCA
    trainPCA,testPCA = split_for_reg(data_pca)
    print("Multiple Linear Regression on data after pca\n")
    mul_lin_reg(trainPCA.copy(),testPCA.copy())
    print("Polynomial curve fitting on data after pca")
    poly_reg(trainPCA.copy(),testPCA.copy())

    corr_lin(data,train,test)
main()
