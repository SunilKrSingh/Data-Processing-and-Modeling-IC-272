import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def read_data(path):
    df = pd.read_csv(path)
    return df

def show_box_plot(attr, df):
    plt.boxplot(df[attr])
    return None
    
def replace_outliers(df):
    cols = [i for i in df]
    for col in cols:
        q3 = df[col].quantile(0.75)
        q1 = df[col].quantile(0.25)
        iqr = q3-q1
        print(q1, q3, iqr)
        median = df[col].median()
        for i in range(len(df)):
            if df[col][i]>=q3+(1.5*iqr):
                df[col][i] = median
            elif df[col][i]<=q1-(1.5*iqr):
                df[col][i] = median
    return df
    
def range1(df, attr):
    mn = df[attr].min()
    mx = df[attr].max()
    return(tuple([mn,mx]))

def min_max_normalization(df, rng1, rng0):
    cols = [i for i in df]
    cols = cols[:-1]
    for col in cols:
        mx = df[col].max()
        mn = df[col].min()
        new_mx = rng1
        new_mn = rng0
        for i in range(len(df)):
            nr=(float(df[col][i])-mn)/(mx-mn)
            df[col][i] = (nr*(new_mx-new_mn))+new_mn
    return df

def standardize(df):
    cols = [i for i in df]
    cols = cols[:-1]
    for col in cols:
        mean = df[col].mean()
        stdDev = df[col].std()
        print("col=",col,"; mean=",mean,"; stdDev=",stdDev)
        for i in range(len(df)):
            df[col][i] = ((df[col][i])-mean)/stdDev
    return df

def main():
    df = read_data("/home/sunil/Desktop/sem3/ds3/data_science_3/lab3/files/winequality_red_original.csv")
    #print(df.head())
    
    cols = [i for i in df]
    """
    for col in cols:
        print(col)
        show_box_plot(col, df)
        plt.show()
    """
    pd.options.mode.chained_assignment = None
    df = replace_outliers(df)
    #print(df.head())
    
    df2=df
    df3=df
    #df4=df
    """
    for col in cols:
        print(col)
        show_box_plot(col, df)
        plt.show()
    """
    for col in cols:
        print(col, range1(df, col))
    
    df = min_max_normalization(df, 1,0)
    print(df.head())
    
    df2 = min_max_normalization(df2, 20,0)
    df2.head()
    
    df3 = standardize(df3)
    df3.head()
    
    cols = [i for i in df3]
    cols = cols[:-1]
    for col in cols:
        mean = df[col].mean()
        stdDev = df[col].std()
        print("col=",col,"; mean=",mean,"; stdDev=",stdDev)
        
    ###################rest by yasho
    mx = MinMaxScaler()  
    df7 = df
    st = StandardScaler()  
    for i in range(len(cols)-1):
        df7[l[i]] = pd.DataFrame(mx.fit_transform(pd.DataFrame(df[l[i]])),columns=[l[i]])
    df8 = df
    for i in range(len(cols)-1):
        df8[l[i]] = pd.DataFrame(st.fit_transform(pd.DataFrame(df[l[i]])),columns=[l[i]])

if __name__=="__main__":
    main()