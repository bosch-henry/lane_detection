import numpy as py
from config import *
from data_process.line_fitting_core import *


# generate random sample point
x = np.random.uniform(-5, 5, size=100)
y = 0.25*x**2 + x + 1 + np.random.normal(0, 1, 100)

scatter = np.array([x, y])
scatter = np.transpose(scatter)
#print(scatter)
#print(np.shape(scatter))

line_OLS = LineFitting(scatter, method="OLS", degree=3)
line_OLS.line_regression()
line_OLS_matrix_y_pred = line_OLS.get_regression_matrix()
print(line_OLS_matrix_y_pred)
line_OLS_coef = line_OLS.get_coef()
print(line_OLS_coef)
line_OLS.line_plotting(color='red', label='OLS')

line_RANSAC = LineFitting(scatter, method="RANSAC", degree=3)
line_RANSAC.line_regression()
line_RANSAC_matrix_y_pred = line_RANSAC.get_regression_matrix()
print(line_RANSAC_matrix_y_pred)
line_RANSAC_coef = line_RANSAC.get_coef()
print(line_RANSAC_coef)
line_RANSAC.line_plotting(color='green', label='RANSAC')


plt.legend(loc='lower right')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()
