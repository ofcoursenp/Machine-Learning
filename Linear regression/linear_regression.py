import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabeties = datasets.load_diabetes()

diabeties_x = diabeties.data[:, np.newaxis, 2]

diabeties_x_train = diabeties_x[:-30]
diabeties_x_test = diabeties_x[-30:]

diabeties_y_train = diabeties.target[:-30]
diabeties_y_test = diabeties.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabeties_x_train, diabeties_y_train)

diabeties_y_predict = model.predict(diabeties_x_test)

print(mean_squared_error(diabeties_y_test, diabeties_y_predict))

plt.scatter(diabeties_x_test, diabeties_y_test)
plt.plot(diabeties_x_test, diabeties_y_predict)

plt.show()