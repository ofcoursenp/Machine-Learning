from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = datasets.load_iris()

x = iris['data'][:,3:]

y = (iris['target'] == 2).astype(int)

# print(y)
# print(x)

clf = LogisticRegression()
clf.fit(x,y)

print(clf.predict([[0.3]]))
