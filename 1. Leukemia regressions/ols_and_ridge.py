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


""" Part A: Build standard linear model on training data and make predictions of the test data. """
ols = linear_model.LinearRegression(fit_intercept=False)
ols.fit(train_x, train_y)
# The mean square error (RSS):
print("Residual sum of squares: %.2f" % np.sum((ols.predict(test_x) - test_y) ** 2))
# Explained variance score (1 is perfect prediction, 0 is that the model is not better than random choice,
# and negative value is that the model is WORST than random prediction.) Unfortunately, using the squared error loss
# for classification also penalized points far in the correct side of the decision boundary. On the contrary,
# the logistic function does not incur in this kind of wrong given penalties.
print('Variance score: %.2f' % ols.score(test_x, test_y))

# Predictions visualization:
axis_x = [i for i in range(0,len(test_y))]
sep_y = [0 for i in range(0,len(test_y))]
plt.plot(axis_x, sep_y, color='black', linewidth=2)
plt.scatter(axis_x, test_y, color='r', alpha=.5, s=100, label='y-test data')
plt.scatter(axis_x, ols.predict(test_x), color='b', marker='+', s=60, label='y-test prediction')
plt.xlabel('samples (patients)')
plt.ylabel('Leukemia Classification -- ALL (down) | AML (up)')
plt.title('Ordinary Least Square Linear Model', fontsize=18, fontweight='bold')
plt.axis('tight')
plt.grid(True)
plt.ylim(-2, 2)
plt.savefig("ols.png")
plt.legend()
plt.show()

""" Part B: Build linear shrinkage model on the training data and make predictions on the test data.
Report results with several lambda values. """
ridge = linear_model.Ridge(fit_intercept=False)
ridge.set_params(alpha=(10 ** 10))
ridge.fit(train_x, train_y)

print("Residual sum of squares: %.2f" % np.sum((ridge.predict(test_x) - test_y) ** 2))
print('Variance score: %.2f' % ridge.score(test_x, test_y))
# Predictions visualization:
axis_x = [i for i in range(0,len(test_y))]
sep_y = [0 for i in range(0,len(test_y))]
plt.plot(axis_x, sep_y, color='black', linewidth=2)
plt.scatter(axis_x, test_y, color='r', alpha=.5, s=100, label='y-test data')
plt.scatter(axis_x, ridge.predict(test_x), color='b', marker='+', s=60, label='y-test prediction')
plt.xlabel('samples (patients)')
plt.ylabel('Leukemia Classification -- ALL (down) | AML (up)')
plt.title('Ridge Regression | Lambda = 10^10', fontsize=18, fontweight='bold')        # Predictions visualization over the test data
plt.axis('tight')
plt.grid(True)
plt.ylim(-2, 2)
plt.legend()
plt.show()

# Using various Lambdas:
ridge1 = linear_model.Ridge(fit_intercept=False)
ridge1.set_params(alpha=(.5))
ridge1.fit(train_x, train_y)
ridge2 = linear_model.Ridge(fit_intercept=False)
ridge2.set_params(alpha=(10 ** 3))
ridge2.fit(train_x, train_y)
ridge3 = linear_model.Ridge(fit_intercept=False)
ridge3.set_params(alpha=(10 ** 9))
ridge3.fit(train_x, train_y)
ridge4 = linear_model.Ridge(fit_intercept=False)
ridge4.set_params(alpha=(10 ** 10))
ridge4.fit(train_x, train_y)
axis_x = [i for i in range(0,len(test_y))]
sep_y = [0 for i in range(0,len(test_y))]

fig, axarr = plt.subplots(2, 2)
fig.suptitle('Ridge Regression', fontsize=18, fontweight='bold')

axarr[0, 0].plot(axis_x, sep_y, color='black', linewidth=1)
axarr[0, 0].scatter(axis_x, test_y, color='r', alpha=.5, s=100, label='y-test data')
axarr[0, 0].scatter(axis_x, ridge1.predict(test_x), color='b', marker='+', s=60, label='y-test prediction')
axarr[0, 0].set_title('Lambda = 0.5', fontweight='bold')
axarr[0, 1].plot(axis_x, sep_y, color='black', linewidth=1)
axarr[0, 1].scatter(axis_x, test_y, color='r', alpha=.5, s=100, label='y-test data')
axarr[0, 1].scatter(axis_x, ridge2.predict(test_x), color='b', marker='+', s=60, label='y-test prediction')
axarr[0, 1].set_title('Lambda = 10^3', fontweight='bold')
axarr[1, 0].plot(axis_x, sep_y, color='black', linewidth=1)
axarr[1, 0].scatter(axis_x, test_y, color='r', alpha=.5, s=100, label='y-test data')
axarr[1, 0].scatter(axis_x, ridge3.predict(test_x), color='b', marker='+', s=60, label='y-test prediction')
axarr[1, 0].set_title('Lambda = 10^9', fontweight='bold')
axarr[1, 1].plot(axis_x, sep_y, color='black', linewidth=1)
axarr[1, 1].scatter(axis_x, test_y, color='r', alpha=.5, s=100, label='y-test data')
axarr[1, 1].scatter(axis_x, ridge4.predict(test_x), color='b', marker='+', s=60, label='y-test prediction')
axarr[1, 1].set_title('Lambda = 10^10', fontweight='bold')

for i in range(0, 2):
    for j in range(0, 2):
        axarr[i, j].set_xlabel('samples (patients)')
        axarr[i, j].set_ylabel('Leukemia Classification -- ALL (down) | AML (up)')
        axarr[i, j].grid(True)
        axarr[i, j].set_ylim([-2, 2])
        axarr[i, j].legend(loc='upper left', framealpha=0.6)
        # axarr[i, j].set_xlim(xmin=-5)
fig.subplots_adjust(hspace=.25)
plt.show()

""" Part C Cross-validation to set lambda/alpha. """
ridge_cv = linear_model.RidgeCV(alphas=[0.5, 10**3, 10**9, 10**10], fit_intercept=False)
ridge_cv.fit(train_x, train_y)
print(ridge_cv.alpha_)