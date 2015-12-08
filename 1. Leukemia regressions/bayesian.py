__author__ = 'Juxtux'

from fns import *
from sklearn import linear_model
import matplotlib.pyplot as plt

# Step1: Treatment of data with Python methods. The math comes afterwards.
train_f = open("train.txt", "r")
train_x = file_to_matrix(train_f)
train_f.close()

test_f = open("test.txt", "r")
test_x = file_to_matrix(test_f)
test_f.close()

samples_f = open("samples.txt", "r")
y = file_to_y(samples_f)
samples_f.close()

y = y_array(y, 'ALL', 'AML')
train_y = y[0:38]
test_y = y[38:]

h_alpha, h_lambda = (1.e-10), (1.e-6)
b_reg = linear_model.BayesianRidge(fit_intercept=False, compute_score=False,
                                   alpha_1=h_alpha, alpha_2=h_alpha, lambda_1=h_lambda, lambda_2=h_lambda)
b_reg.fit(train_x, train_y)

print("Residual sum of squares: %.2f" % np.mean((b_reg.predict(test_x) - test_y) ** 2))
print('Variance score: %.2f' % b_reg.score(test_x, test_y))

axis_x = [i for i in range(0, len(test_y))]
sep_y = [0 for i in range(0, len(test_y))]
plt.plot(axis_x, sep_y, color='black', linewidth=2)
plt.scatter(axis_x, test_y, color='r', alpha=.5, s=100, label='y-test data')
plt.scatter(axis_x, b_reg.predict(test_x), color='b', marker='+', s=60, label='y-test prediction')
plt.xlabel('samples (patients)')
plt.ylabel('Leukemia Classification -- ALL (down) | AML (up)')
plt.title('Bayesian Regression', fontweight='bold')        # Predictions visualization over the test data
plt.axis('tight')
plt.grid(True)
plt.ylim(-2, 2)
plt.legend(loc='upper left')
plt.show()