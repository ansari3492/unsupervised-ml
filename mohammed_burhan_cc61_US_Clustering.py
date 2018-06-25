# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:12:54 2018

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("tshirts.csv")
features=data.iloc[:,[1,2]].values

#clustering
from sklearn.cluster import KMeans



kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(features)


#visualizing
plt.scatter(features[y_kmeans==0,0],features[y_kmeans==0,1],s=100,c='red',label='medium')
plt.scatter(features[y_kmeans==1,0],features[y_kmeans==1,1],s=100,c='green',label='large')
plt.scatter(features[y_kmeans==2,0],features[y_kmeans==2,1],s=100,c='blue',label='small')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='pink',label='centroid')
plt.title("small vs medium vs large")
plt.xlabel("height")
plt.ylabel("weight")
plt.legend()
plt.show()
center1=kmeans.cluster_centers_[:,0]
centre2=kmeans.cluster_centers_[:,1]