import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(datafile)
X = features
y = targets

# SPLIT TRAIN AND TEST
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_seed)

# FEATURE SCALING
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(    X_test )
'''

# SIMPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X_train, y_train)

y_pred = lin_regressor.predict(X_test)

# PLOTTING RESULTS

plt.scatter(X_train, y_train, color = 'red')
plt.plot(   X_train, lin_regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(   X_train, lin_regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



# POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# PREDICT
lin_reg_2.predict( poly_reg.fit_transform(X_test))

# VISUALISING
plt.scatter(X_test, y_test, color = 'red')
plt.plot(   X_train, lin_reg_2.predict(poly_reg.fit_transform(X_train)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()








# SUPPORT VECTOR REGRESSION
from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(X,y)

# PREDICT
y_pred = svr_reg.predict(X_test)

# VISUALISING
plt.scatter(X_test, y_test, color = 'red')
plt.plot(   X_train, lin_reg_2.predict(X_train), color = 'blue')
plt.title('SVR Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()



# ---------------- DISCRETE ------------------------


# DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
dectree_reg = DecisionTreeRegressor(random_state = 0)
dectree_reg.fit(X,y)

# PREDICT
y_pred = dectree_reg.predict(X_test)

# VISUALISING
plt.scatter(X_test, y_test, color = 'red')

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.plot(   X_grid, dectree_reg.predict(X_grid), color = 'blue')
plt.title('DECISION TREE Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()





# RANDOM FOREST REGRESSION
from sklearn.tree import RandomForestRegressor
randfor_reg = RandomForestRegressor(n_estimators = 100, random_state = 0)
randfor_reg.fit(X,y)

# PREDICT
y_pred = randfor_reg.predict(X_test)

# VISUALISING
plt.scatter(X_test, y_test, color = 'red')

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.plot(   X_grid, randfor_reg.predict(X_grid), color = 'blue')
plt.title('RANDOM FOREST Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

