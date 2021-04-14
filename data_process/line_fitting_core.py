import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import config as cfg


class LineFitting():
    def __init__(self, scatter=None, method="OLS", degree=2):
        # initialize scatter array for line fitting
        self.x = scatter[ : , 0]
        self.y = scatter[ : , 1]
        self.method = method
        self.linear_setting = cfg.GetLineFittingSetting(method, degree)
        self.y_predict = []
        self.coef = []


    def line_regression(self):
        X = self.x.reshape(-1, 1)

        # OLS linear regression method
        if self.method == "OLS":
            # linear regression process
            poly = PolynomialFeatures(degree = self.linear_setting["OLS_degree"]).fit(X)
            X2 = poly.transform(X)
            linear_two  = linear_model.LinearRegression().fit(X2,  self.y)
            y_predict_two = linear_two.predict(X2)
            # set y_predict to class varible
            self.y_predict = y_predict_two
            # set coef to class varible
            coef = []
            coef.append(linear_two.intercept_)
            for i in range(len(linear_two.coef_)):
                coef.append(linear_two.coef_[i])
            self.coef = coef
        
        # RANSAC linear regression method
        elif self.method == "RANSAC":
            # deel with X input with given regression degree
            X4 = []
            for i in range(self.linear_setting["RANSAC_degree"]):
                X4.append(X ** (i + 1))
            X4 = np.hstack(X4)
            # setup ransac regression model
            ransac = linear_model.RANSACRegressor(  max_trials=self.linear_setting["RANSAC_max_trials"],
                                                    min_samples=self.linear_setting["RANSAC_min_samples"],
                                                    loss=self.linear_setting["RANSAC_loss"],
                                                    residual_threshold=self.linear_setting["RANSAC_residual_threshold"],
                                                    stop_probability=self.linear_setting["RANSAC_stop_probability"] )
            ransac.fit(X4, self.y)
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)            
            line_y_ransac = ransac.predict(X4)
            #set y_predict to class varible
            self.y_predict = line_y_ransac
            #set coef to class varible
            coef = []
            coef.append(ransac.estimator_.intercept_)
            for i in range(len(ransac.estimator_.coef_)):
                coef.append(ransac.estimator_.coef_[i])
            self.coef = coef

        else:
            None


    # get the linear regression line coefficient list
    def get_coef(self):
        return np.array(self.coef).reshape(-1,1)


    # get the linear regression line y_predict (according to x input order)
    def get_y_predict(self):
        return self.y_predict


    def get_regression_matrix(self):
        regression_matrix = np.array([self.x, self.y, self.y_predict])
        regression_matrix = np.transpose(regression_matrix)
        return regression_matrix


    # plot the input scatter and regression line
    def line_plotting(self, color, label):
        plt.scatter(self.x, self.y, label='sample points')
        plt.plot(np.sort(self.x), self.y_predict[np.argsort(self.x)], color = color, label = label)
        


'''
# generate random sample point
x = np.random.uniform(-5, 5, size=100)
X = x.reshape(-1, 1)
y = 0.25*x**2 + x + 1 + np.random.normal(0, 1, 100)
# plot original sample points
#plt.scatter(x, y)
#plt.show()

# normal polynominal regression
poly = PolynomialFeatures(degree = 2).fit(X)
X2 = poly.transform(X)
#print(X2)
linear_two  = linear_model.LinearRegression().fit(X2,  y)
y_predict_two = linear_two.predict(X2)
# plot polynominal regression result
#plt.scatter(x, y)
#plt.plot(np.sort(x), y_predict_two[np.argsort(x)], color =  'red')
#plt.show()
print(y_predict_two)
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
print(line_y_ransac)
print('RANSAC intercept = {0}'.format(ransac.estimator_.intercept_))
print('RANSAC coefficient: c1 = {0}, c2 = {1}, c3 = {2}'.format(ransac.estimator_.coef_[0], ransac.estimator_.coef_[1], ransac.estimator_.coef_[2]))

plt.scatter(x, y, label='sample points')
plt.plot(np.sort(x), y_predict_two[np.argsort(x)], color = 'yellow', label = 'Poly regressor')
plt.plot(np.sort(x), line_y_ransac[np.argsort(x)], color = 'green', label = 'RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()
'''
