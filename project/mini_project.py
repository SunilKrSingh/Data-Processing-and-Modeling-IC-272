"""
IC-272
Mini-Project

Thursday Batch
Group 8

AUTHORS:
Abhinav Kumar (B18099)
Rupanshi Saini (B18136)
Sunil Kumar Singh (B18026)
Anjali Deep (B18158)
Deepti Singh (B18110)
Sachit Batra (B18137)

TOPIC: Gaining Insights from different performance measures(PMs) of BNG Devices. 

Description of Dataset:

The dataset contained following independent attributes to consider:
    CreationTime: Date and time of the recording of sample
    AuthenticateCount: Number of active subscribers authenticated their connection.  
    ActiveCount: Number of active subscribers connected to the device.
    DisconnectCount: Number of active subscribers disconnected from the device.
    CPUUtil: contains the % of usage of processor in the device.
    MemoryUsed: Total memory in Bytes used in the device
    MemoryFree: Total of memory in Bytes free in the device
    TempMin: Minimum temperatures among the temperatures recorded from the different slots in the device
    TempMax: Maximum temperatures among the temperatures recorded from the different slots in the device
    TempAvg: Average temperatures among the temperatures recorded from the different slots in the device
    OutBandwidth: Total bandwidth utilization in Bytes from the output ports of all the interfaces.
    InTotalPPS: Total packets per second transmitted from the input ports of all the interfaces.
    OutTotalPPS: Total packets per second transmitted from the output ports of all the interfaces.

Dependent Attribute: InBandwidth

Predictive Analytics: Regressive Analysis

"""


#importing required libraries
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
from statsmodels.tsa.ar_model import AR
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm

#Function to read .csv file
def read_data(path):
    return(pd.read_csv(path))

#Function to show informations of the data
def info(df):
    print(df.describe())

def boxplt(data):
    plt.boxplot(data)

#Function to print accuracy using RMSE Values
def prediction_accuracy(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# first lets start with preprocessing of data
def check_null_val(data):
    print("\nNull values in different attributes: ")
    print(data.isnull().sum())
    ##there is no null values present
    
###############################################################
#Function to  fijnd counts of outliers
def outlier_detection(df):
    y = list(df.columns)
    y.remove("CreationTime") ##it is a categorical data
    z = 0
    for i in y:
        s=list(df[i])
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

#Function to  plot boxplot of the data
def boxplt(data):
    plt.boxplot(data)

def boxplot(norma_df,norma_df_wtdout):  ## pass both normalised actual and outlier removed
    norma_df.boxplot(vert=True,rot=45)
    plt.show()
    norma_df_wtdout.boxplot(vert=True,rot=45)
    plt.show()

###############################################################
#Function to Z-normalise the data
def standardization(data):
    df1 = data.copy()
    x = list(df1.columns)
    x.remove("CreationTime")
    for i in x:
        df1[i] = ((df1[i] - df1[i].mean())/df1[i].std())
    return df1
###############################################################
    #Function to min-max normalise the data
def normalise(data,newmax,newmin):
    df1 = data.copy()
    x = list(data.columns)
    x.remove("CreationTime")
    for i in x:
        df1[i] = (((df1[i] - df1[i].min())/(df1[i].max() - df1[i].min()))*(newmax-newmin))+newmin
    return df1
###############################################################
#Function to replace outliers with median
def replace_outlier(data):
    df1 = data.copy()
    y = list(df1.columns)
    y.remove("CreationTime")     ##it is a categorical data
    for i in y:
        s=(df1[i])
        q1=df1[i].quantile(0.25)
        q3=df1[i].quantile(0.75)
        iqr=q3-q1
        c=df1[i].median()
        for j in range(len(df1[i])):
            if (q1-1.5*iqr)>=s[j] or s[j]>=(q3+1.5*iqr):
                df1.loc[j,i]=c
    return df1
##############################################################
#Function to do analysis on the basis of correlation values
def corr_analysis(data):
    col = list(data.columns)
    col.remove("CreationTime")
    col.remove( "InBandwidth")
    target_attribute = "InBandwidth"
    cor = []
    for i in col:
        corval = abs(st.pearsonr(data[target_attribute],data[i])[0])
        cor.append(corval)
    return cor

def print_corr(data):
    col = list(data.columns)
    col.remove("CreationTime")
    col.remove( "InBandwidth")
    target_attribute = "InBandwidth"
    cor = []
    print("\nCorrelation Coefficient of 'InBandwidth' with ")
    for i in col:
        corval = abs(st.pearsonr(data[target_attribute],data[i])[0])
        cor.append(corval)
        print(i," = ",corval)

###############################################################
#Function to apply PCA for i reduced attributes
def PcA(data,i):
    df1 = data.copy()
    df1.drop(columns = ["CreationTime"], inplace = True)
    df1.drop(columns = ["InBandwidth"], inplace = True)
    pca = PCA(n_components=i)
    newdata = pd.DataFrame(pca.fit_transform(df1))
    newdata2=pd.concat([data[['CreationTime']],newdata,data[['InBandwidth']]], axis = 1)
    return newdata2
###############################################################
#Function to remove irrelavant features
def feature_selection(data):
    col = list(data.columns)
    col.remove("CreationTime")
    col.remove( "InBandwidth")
    #remove the attributes with more than 50% missing values
    cor=(corr_analysis(data))
    plt.figure(figsize=(12,10))
    corX = data.corr()
    ss.heatmap(corX, annot=True, cmap=plt.cm.Reds)
    plt.show()
    irrelevant_features=[]
    for i in range(len(col)):
        if(cor[i]<0.1):
            irrelevant_features.append(col[i])
    for i in range(len(irrelevant_features)):
        data.drop(irrelevant_features[i],1)
    print("Features removable: ")
    for i in range(len(irrelevant_features)):
        print(i+1, " ", irrelevant_features[i])
    return data

#####end of data preprocessing


###data predictive analysis 
## inbandwidth is dependent attribute
#Function to split data into test and training data
def split_for_reg(df):
    train,test = train_test_split(df,train_size = 0.70,test_size = 0.30,random_state= 42,shuffle = True)
    return train,test

#Function to apply multiple linear regression
def mul_lin_reg(train,test):
    print("Multiple Linear regression\n")
    ### categorical data removal
    train.drop(columns = ["CreationTime"], inplace = True)
    test.drop(columns = ["CreationTime"], inplace = True)
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
    print(metrics.r2_score(actual,pred))
    print("rmse of train data : ",prediction_accuracy(Y,pred_train),'\n')
    print(metrics.r2_score(actual,pred))
    plt.scatter(actual,pred , color = "blue",  label = 'Test data')
    plt.xlabel('original InBandwidth')
    plt.ylabel('predicted InBandwidth')
    plt.show()
#############################################################

#Function to perform polynomial regression and predict values.
def poly_reg(trainX,testX):
    ### categorical data removal
    print("Polynomial Curve Fitting\n")
    train=trainX.drop("CreationTime", axis=1)
    test=testX.drop("CreationTime", axis=1)
    
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

def poly_on_range(trainX,testX,start, end):
    ### categorical data removal
    print("Polynomial Curve Fitting\n")
    train=trainX.drop("CreationTime", axis=1)
    test=testX.drop("CreationTime", axis=1)

    Y = train["InBandwidth"]
    train.drop(columns = ["InBandwidth"],inplace =True)
    actual = test["InBandwidth"]
    test.drop(columns = ["InBandwidth"],inplace =True)
    Range = [i for i in range(start, end)]
    RMSEs_onTest, RMSEs_onTrain = [], []
    for i in Range:
        poly = PolynomialFeatures(degree = i)
        new_x = poly.fit_transform(train)
        new_test = poly.fit_transform(test)
        reg = LinearRegression()
        reg.fit(new_x,Y)
        pred = reg.predict(new_test)
        RMSEs_onTest.append(prediction_accuracy(actual,pred))
        print("rmse test : ",prediction_accuracy(actual,pred))
        print(metrics.r2_score(actual,pred))
        RMSEs_onTrain.append(prediction_accuracy(Y,reg.predict(new_x)))
        print("rmse train : ",prediction_accuracy(Y,reg.predict(new_x)),'\n')
        print(metrics.r2_score(Y,reg.predict(new_x)))
    plt.plot(Range, RMSEs_onTest)
    plt.title("RMSE vs. Degree of Polynomial(Test Data)")
    plt.ylabel("RMSE")
    plt.xlabel("Degree of Polynomial Regression")
    plt.show()
    
    plt.plot(Range, RMSEs_onTrain)
    plt.title("RMS vs. Degree of Polynomial(Training Data)")
    plt.ylabel("RMSE")
    plt.xlabel("Degree of Polynomial Regression")
    plt.show()
    

def corr_column(df):
    col = list(df.columns)
    col.remove("CreationTime")
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

def auto_regression(df):
    train,test= split_for_reg(df['InBandwidth'])
    model=AR(train)
    model_fit=model.fit()
    print('lag:',model_fit.k_ar)
    predict_ar=model_fit.predict(start=len(train),end=len(train)+len(test)-1,dynamic=False)
    error=prediction_accuracy(test,predict_ar)
    print("rmse",error)

def plot_acf(df):
    data=df[['InBandwidth']]
    #plt.plot(data)
    #plt.show()

    print("acf plot")
    sm.graphics.tsa.plot_acf(data,lags=40)
    plt.show()



def main():
    #Reading .csv file
    path = "D:\sem3\ds3\data_science_3\project\group8.csv"
    data = pd.read_csv(path)
    
    #=================== descriptive analysis ===================#
    
    #Checking null values and print outliers on original data
    check_null_val(data)
    print("\nOutlier counts before replacing outliers:")
    outlier_detection(data)

    #Checking null values and print outliers after outliers removal
    df1 = replace_outlier(data)
    print("\nOutlier counts after replacing outliers:")
    outlier_detection(df1)
    info(df1)
    
    #================== pre-processing on data ==================#
    
    #Operationns on outlier removed dataframe
    stand_df = standardization(df1)     #Z-normalised data
    norma_df = normalise(df1,1,0)       #min-max-normalised data
    norma_df_wtdout=normalise(df1,1,0) ### normalise after outlier removal
    data_pca = PcA(data,2)              #PCA applied data
    print_corr(data)                    #Correlation analysis with target as inbandwidth column
    data_FS=feature_selection(data)     #Feature selection on data
    boxplot(norma_df,norma_df_wtdout)
    pd.plotting.lag_plot(data["InBandwidth"],marker = "*")
    plt.show()

    r = pd.DataFrame(pd.to_datetime(data["CreationTime"]))
    r.columns = ["CreationTime"]
    r['new_date'] = [d.date() for d in r["CreationTime"]]
    r['new_time'] = [d.time() for d in r["CreationTime"]]
    r["InTotalPPS"]=data["InTotalPPS"]
    df = r.groupby('new_date')
    dat = df.get_group('2018-10-12')
    #print(dat)
    plt.plot(dat["new_time"],dat["InTotalPPS"])
    plt.title("variation of active users on 2018-10-12 ")
    plt.xlabel("time variation ")
    plt.ylabel("ActiveCount")
    plt.show()
    #================= data predictive analysis =================#

    #results with original datasets when replaced with outliers
    train,test = split_for_reg(data)
    print("\nMultiple Linear Regression on untreated data\n")
    mul_lin_reg(train.copy(),test.copy())
    print("Polynomial curve fitting on untreated data")
    poly_on_range(train.copy(),test.copy(),2,5)

    #results with preprocessed datasets when replaced with outliers
    trainP,testP = split_for_reg(df1)
    print("\nMultiple Linear Regression on data without outliers\n")
    mul_lin_reg(trainP.copy(),testP.copy())
    print("Polynomial curve fitting on data without outliers\n")
    poly_on_range(trainP.copy(),testP.copy(),2,5)

    #resuls on standardized data
    trainS,testS = split_for_reg(stand_df)
    print("\nMultiple Linear Regression on standardized data\n")
    mul_lin_reg(trainS.copy(),testS.copy())
    print("Polynomial curve fitting on standardized data")
    poly_on_range(trainS.copy(),testS.copy(),2,5)

    #results on normalized data
    trainN,testN = split_for_reg(norma_df)
    print("\nMultiple Linear Regression on normalized data\n")
    mul_lin_reg(trainN.copy(),testN.copy())
    print("Polynomial curve fitting on normalized data")
    poly_on_range(trainN.copy(),testN.copy(),2,5)

    #results after feature selection
    trainFS,testFS = split_for_reg(data_FS)
    print("\nMultiple Linear Regression on data after feature selection\n")
    mul_lin_reg(trainFS.copy(),testFS.copy())
    print("Polynomial curve fitting on data after feature selection")
    poly_on_range(trainFS.copy(),testFS.copy(),2,5)

    #results after PCA
    trainPCA,testPCA = split_for_reg(data_pca)
    print("\nMultiple Linear Regression on data after pca\n")
    mul_lin_reg(trainPCA.copy(),testPCA.copy())
    print("Polynomial curve fitting on data after pca")
    poly_on_range(trainPCA.copy(),testPCA.copy(),2,5)

    corr_lin(data,train,test)

    #resuls after Auto regression
    print("Auto regression on original data")
    auto_regression(data) #on original data
    plot_acf(data)
    print("\nAuto regression on standardized data")
    auto_regression(stand_df) #on standardized data
    plot_acf(stand_df)
    print("\nAuto regression on normalized data")
    auto_regression(norma_df) #on normalized data
    plot_acf(norma_df)
    print("\nAuto regression on data after outlier removal")
    auto_regression(df1) #on data after replacing qutlier
    plot_acf(df1)
    print("\nAuto regression on data after feature selection")
    auto_regression(data_FS) #on data after feature selection
    plot_acf(data_FS)


if __name__ == "__main__":
    main()
