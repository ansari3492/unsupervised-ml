# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:13:29 2018

@author: Lenovo
"""
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("deliveryfleet.csv")
features=data.iloc[:,[1,2]].values

#clustering
from sklearn.cluster import KMeans


#devide in urban  and rural area
kmeans=KMeans(n_clusters=2,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(features)


#visualizing
plt.scatter(features[y_kmeans==0,0],features[y_kmeans==0,1],s=100,c='red',label='urban')
plt.scatter(features[y_kmeans==1,0],features[y_kmeans==1,1],s=100,c='green',label='rural')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='pink',label='centroid')
plt.title("urban vs rural")
plt.xlabel("Distance")
plt.ylabel("Speed")
plt.legend()
plt.show()


#devide according to speed in urban and rural
kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(features)


#visualizing
plt.scatter(features[y_kmeans==0,0],features[y_kmeans==0,1],s=100,c='red',label='normal urban')
plt.scatter(features[y_kmeans==1,0],features[y_kmeans==1,1],s=100,c='green',label='normal rural')
plt.scatter(features[y_kmeans==2,0],features[y_kmeans==2,1],s=100,c='blue',label='over rural')
plt.scatter(features[y_kmeans==3,0],features[y_kmeans==3,1],s=100,c='yellow',label='over urban')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='pink',label='centroid')
plt.title("urban vs rural")
plt.xlabel("Distance")
plt.ylabel("Speed")
plt.legend()
plt.show()








