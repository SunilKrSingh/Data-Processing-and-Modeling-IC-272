from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as agg
from sklearn.cluster import DBSCAN
import pandas as pd

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix,axis=0))/np.sum(contingency_matrix)

df = pd.read_csv("D:\sem3\ds3\data_science_3\lab11\inLab\Iris.csv")
x = list(df["Species"])
df1 = df.iloc[:,1:5]

pca = PCA(n_components=4).fit(df1)
reduced_data = PCA(n_components=2).fit_transform(df1)

X, Y = zip(*reduced_data)
plt.scatter(X,Y)

agg_clustering = agg(n_clusters=3).fit(reduced_data)
plt.scatter(X, Y, c = agg_clustering.labels_)
plt.title("Agglomerative_clustering_model")
plt.show()
print("Agglomerative_clustering_model purity score is ", purity_score(x, agg_clustering.labels_))


print("DBCAN_clustering_model ##################################################33")
EPS = [0.05, 0.5, 0.95]
MIN_SAMPLES = [1, 5,10,20]
for eps_ in EPS:
    for min_ in MIN_SAMPLES:
        dbscan_clustering = DBSCAN(eps=eps_, min_samples=min_).fit(reduced_data)
        plt.scatter(X, Y, c = dbscan_clustering.labels_)
        plt.title(f"eps = {eps_}, min_samples = {min_}")
        plt.show()
        print(f"(eps = {eps_}, min_samples = {min_})", "==>",purity_score(x, dbscan_clustering.labels_))