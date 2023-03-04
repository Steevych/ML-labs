#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 23:19:33 2023

@author: Steeve
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay

iris=load_iris()
iris.feature_names
#print(iris.feature_names)
#print(iris.data[0:5,:])
#print(iris.target[0:5])
#print(iris.data)


X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#print(X_train.shape)
#print(X_test.shape)

SVMmodel=SVC(kernel='linear')
classifier = SVMmodel=SVC(kernel='linear', C=10).fit(X_train,y_train)
SVMmodel.get_params()
SVMmodel.score(X_test,y_test)


#decision_function = np.dot(X, classifier.coef_[0]) + classifier.intercept_[0]
decision_function = classifier.decision_function(X_train)
supvectors=SVMmodel.support_vectors_


sns.scatterplot(
    x=X_train[:, 0],
    y=X_train[:, 1],
    hue=y_train
)

"""ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    classifier,
    X_train,
    ax=ax,
    grid_resolution=50,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
)
plt.scatter(
    supvectors[:, 0],
    supvectors[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)


plt.xlabel("x")
plt.ylabel("y")
plt.show()"""

"Anomaly detection"

from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random

random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(4, 4))

plt.scatter(x[:,0], x[:,1])
plt.show()

SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)


SVMmodelOne.fit(x)
pred = SVMmodelOne.predict(x)
anom_index = where(pred==-1)
values = x[anom_index]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.axis('equal')
#plt.show()

plt.scatter(
    supvectors[:, 0],
    supvectors[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)

ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    SVMmodelOne,
    x,
    ax=ax,
    grid_resolution=50,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
)
    
plt.tight_layout()
plt.axis('equal')
plt.show()
