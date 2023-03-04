# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"PCA through Singular Value Decomposition"
import numpy as np
# Defined 3 points in 2D-space:
X=np.array([[2, 1, 0],[4, 3, 0]])
# Calculate the covariance matrix:
# R = np.cov(X)

R=np.matmul(X,X.T)/3
#Raltern=X@np.transpose(X)/3
#print(Ralten)
#print(R)


[U,D,V]=np.linalg.svd(R)  # call SVD decomposition
u1=U[:,0] # new basis vectors
u2=U[:,1]

Xi1=np.matmul(np.transpose(X),u1)
Xi2=np.matmul(np.transpose(X),u2)
#print(Xi1)
#print(Xi2)

Xaprox=np.matmul(u1[:,None],Xi1[None,:])+np.matmul(u2[:,None],Xi2[None,:])
print(Xaprox)

from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris




"PCA on IRIS data"
iris=load_iris()
iris.feature_names

X=iris.data
y=iris.target
#print(iris.feature_names)
#print(iris.data[0:5,:])
#print(iris.target[:])

Xscaler = StandardScaler()
Xpp=Xscaler.fit_transform(X)

#print(np.mean(Xpp[:,0]))
#print(np.std(Xpp[:,0]))

pca = PCA(n_components=3)
pca.fit(Xpp)
Xpca=pca.transform(Xpp)
#print(pca.getcoverlance())
# you can plot the transformed feature space in 3D:
#axes2=plt.axes(projection='3d')
#axes2.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
#axes2.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
#axes2.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta') """

with plt.style.context('ggplot'):
     plt.figure(figsize=(6, 4))

     plt.bar(range(3), pca.explained_variance_, alpha=0.5, align='center',
     label='individual explained variance')
     plt.ylabel('Explained variance ratio')
     plt.xlabel('Principal components')
     plt.xticks(range(3))
     plt.legend()
     plt.tight_layout()
     #plt.show()

plot = plt.scatter(Xpca[:,0], Xpca[:,1], c=y)
plt.legend(handles=plot.legend_elements()[0], labels=list(iris.feature_names))
#plt.show()

"KNN classifier"

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_train, y_train)
Ypred=knn1.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, Ypred, ax=axs[0])




pca = PCA(n_components=2)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X,y,test_size=0.3)


knn2=KNeighborsClassifier(n_neighbors = 3)
knn2.fit(X_train_pca, y_train_pca)
Ypred_pca=knn2.predict(X_test_pca)
ConfusionMatrixDisplay.from_predictions(y_test_pca, Ypred_pca, ax=axs[1])

x=iris.data[:, :2]
y=iris.target
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X,y,test_size=0.3)
knn3=KNeighborsClassifier(n_neighbors = 3)
knn3.fit(X_train_2d, y_train_2d)
Ypred=knn3.predict(X_test_2d)
ConfusionMatrixDisplay.from_predictions(y_test_2d, Ypred, ax=axs[2])
axs[0].title.set_text('Full Dataset')
axs[1].title.set_text('PCA with 2 components')
axs[2].title.set_text('2 first columns')
fig.tight_layout()
plt.show()
