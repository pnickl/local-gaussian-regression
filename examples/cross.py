import numpy as np

import scipy as sc
from scipy import stats

from lgr.options import Options
from lgr.batchLGR.lgr import LGR

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


N_train = 2000
D_in = 2

seed = 411
np.random.seed(seed)


def cross_2d(x1, x2, test):
    # definition of cross function from Vijayakumar et al.
    # Locally Weighted Projection Regression (2000), ICML
    import scipy as sc

    f1 = np.exp(-10 * x1 ** 2)
    f2 = np.exp(-50 * x2 ** 2)
    f3 = 1.25 * np.exp(-5 * (x1 ** 2 + x2 ** 2))
    noise = sc.stats.norm.rvs(np.zeros(len(f1)), 0.01 * np.ones(len(f1)), len(f1))
    f_ = np.maximum(f1, f2)
    y = np.maximum(f_, f3) + noise

    if test == True:
        x1 = np.reshape(x1, -1)
        x2 = np.reshape(x2, -1)
    X = np.vstack((x1, x2)).T

    return X, y


def load_cross_data(N_train):

    # 2000 uniformly distributed training inputs with zero mean gaussian noise of 0.2 standard deviation
    X1_train = np.random.uniform(-1, 1, N_train) + np.random.normal(0, 0.2, N_train)
    X2_train = np.random.uniform(-1, 1, N_train) + np.random.normal(0, 0.2, N_train)

    # test set is regular 40x40 grid without noise
    x1_linspace = np.linspace(-1, 1, 40)
    x2_linspace = np.linspace(-1, 1, 40)
    X1_test, X2_test = np.meshgrid(x1_linspace, x2_linspace)
    N_test = X1_test.shape[0] * X2_test.shape[0]

    # data
    train_input, train_target = cross_2d(X1_train, X2_train, False)
    test_input, test_target = cross_2d(X1_test, X2_test, True)

    return train_input, train_target, test_input, test_target, N_test


opt = Options(D_in)
opt.activ_thresh = 0.5
opt.max_num_lm = 100
opt.max_iter = 10000
opt.init_lambda = 0.3

opt.print_options()


X_train, Y_train, X_test, Y_test, N_test = load_cross_data(N_train)
Y_train, Y_test = np.reshape(Y_train, (N_train, 1)), np.reshape(Y_test, (N_test, 1))

model = LGR(opt, D_in)
debug = True

model.initialize_local_models(X_train)
initial_local_models = model.get_local_model_activations(X_train)

nmse = model.run(X_train, Y_train, opt.max_iter, debug)
print("TRAIN - NSME: {}".format(nmse[-1]))


Yp = model.predict(X_test)
final_local_models = model.get_local_model_activations(X_train)
number_local_models = final_local_models.shape[1]

print('# test data, final # local models:', final_local_models.shape)

test_mse = mean_squared_error(Y_test, Yp)
test_smse = 1. - r2_score(Y_test, Yp, multioutput='variance_weighted')
print('TEST - MSE:', test_mse, 'NSMSE:', test_smse)

# # plot learned cross function
x1_linspace = np.arange(-1, 1, 0.05)
x2_linspace = np.arange(-1, 1, 0.05)
X1_test, X2_test = np.meshgrid(x1_linspace, x2_linspace)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1_test, X2_test, Yp.reshape(X1_test.shape), 100)
plt.show()
