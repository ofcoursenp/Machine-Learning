from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris = datasets.load_iris()
features = iris.data
labels = iris.target


# for index,i in enumerate(features):
#     print(i)
#     print(labels[index])

#     if index == 10:
#         break


# Tranining 
clf = KNeighborsClassifier()
clf.fit(features,labels)

print(clf.predict(np.array([[1,2,3,4]])))


# Spliting traning 


# iris = datasets.load_iris()

# # print(iris.DESCR)

# features = iris.data[0:150:10]
# labels = iris.target[0:150:10]

# features_training = iris.data[10:150:10]
# labels_traning = iris.target[10:150:10]

# print(features_training,labels_traning)
# print(features,labels)

# # clf = KNeighborsClassifier()
# # clf.fit(features,labels)

# # print()