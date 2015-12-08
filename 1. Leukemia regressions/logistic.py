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

lreg = linear_model.LogisticRegression(penalty='l2', C=1e-5, fit_intercept=False, max_iter=100)
# C: Is the inverse of regularization strength; must be a positive float. Like in support vector machines,
# smaller values specify stronger regularization.
lreg.fit(train_x, train_y)
print("Residual sum of squares: %.2f" % np.sum((lreg.predict(test_x) - test_y) ** 2))
print('Variance score: %.2f' % lreg.score(test_x, test_y))
# Predictions visualization:
axis_x = [i for i in range(0, len(test_y))]
sep_y = [0 for i in range(0, len(test_y))]
plt.plot(axis_x, sep_y, color='black', linewidth=2)
plt.scatter(axis_x, test_y, color='r', alpha=.5, s=100, label='y-test data')
plt.scatter(axis_x, lreg.predict(test_x), color='b', marker='+', s=60, label='y-test prediction')
plt.xlabel('samples (patients)')
plt.ylabel('Leukemia Classification -- ALL (down) | AML (up)')
plt.title('Logistic Regression | Inv Regularization Strength = 1e-5', fontweight='bold')        # Predictions visualization over the test data
plt.axis('tight')
plt.grid(True)
plt.ylim(-2, 2)
plt.legend()
plt.show()
