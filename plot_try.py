import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import config as cfg

# generate random sample point
x = np.random.uniform(-5, 5, size=100)
X = x.reshape(-1, 1)
y = 0.25*x**2 + x + 1 + np.random.normal(0, 1, 100)

# plot original sample points
plt.scatter(x, y, label='sample points')
#plt.show()

# normal polynominal regression
poly = PolynomialFeatures(degree = 2).fit(X)
X2 = poly.transform(X)
#print(X2)
linear_two  = linear_model.LinearRegression().fit(X2,  y)
y_predict_two = linear_two.predict(X2)
# plot polynominal regression result
#plt.scatter(x, y)
plt.plot(np.sort(x), y_predict_two[np.argsort(x)], color =  'red')
#plt.show()
#print(y_predict_two)
print('OLS intercept = {0}'.format(linear_two.intercept_))
print('OLS coefficient: c1 = {0}, c2 = {1}, c3 = {2}'.format(linear_two.coef_[0], linear_two.coef_[1], linear_two.coef_[2]))

# Robustly fit linear model with RANSAC algorithm
X4 = np.hstack([X, X**2, X**3])
#print(X4)
ransac = linear_model.RANSACRegressor()
ransac.fit(X4, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
# Predict data of estimated models
line_y_ransac = ransac.predict(X4)
print(type(line_y_ransac))
print(np.shape(line_y_ransac))
print(line_y_ransac)
print('RANSAC intercept = {0}'.format(ransac.estimator_.intercept_))
print('RANSAC coefficient: c1 = {0}, c2 = {1}, c3 = {2}'.format(ransac.estimator_.coef_[0], ransac.estimator_.coef_[1], ransac.estimator_.coef_[2]))

A = np.array([x, y, line_y_ransac])

print("A = ", A)
print(A.shape)
B = np.transpose(A)
print("B = ", B)
print(B.shape)

#plt.scatter(x, y, label='sample points')
#plt.plot(np.sort(x), y_predict_two[np.argsort(x)], color = 'yellow', label = 'Poly regressor')
plt.plot(np.sort(x), line_y_ransac[np.argsort(x)], color = 'green', label = 'RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.savefig("./plot.jpg", dpi=300)
plt.show()