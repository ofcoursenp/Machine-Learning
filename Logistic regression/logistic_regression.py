from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


# Virginical flower detection

iris = datasets.load_iris()

x = iris['data'][:,3:]

y = (iris['target'] == 2).astype(int)

# print(y)
# print(x)

clf = LogisticRegression()
clf.fit(x,y)

print(clf.predict([[0.3]]))



# Advance Logistic Regression
x_new = np.linspace(0,3,1000).reshape(-1,1)
# print(x_new)

y_prob = clf.predict_proba(x_new)
# print(y_prob)

plt.plot(x_new,y_prob[:,1],'g-',label='virginica')
plt.show()