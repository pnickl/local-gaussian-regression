import numpy as np
from numpy import exp

from lgr.options import Options
from lgr.batchLGR.lgr import LGR

#%%

N = 15000
D = 21
stds = 0.2
seed = 411
np.random.seed(seed)

#%%

def generate_sarcos_data():
    import scipy as sc
    from scipy import io

    # load all available data
    import os
    print(os. getcwd())
    _train_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv.mat')['sarcos_inv']
    _test_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']

    data = np.vstack((_train_data, _test_data))

    train_input = _train_data[:N, :21]
    train_target = _train_data[:N, 21]

    test_input = _test_data[:, :21]
    test_target = _test_data[:, 21]
    return train_input, train_target

# plotting doesn't work for 2D data - but learning does
def generate_2d_data():
    X = (np.random.rand(N, D) - 0.5)*2
    Yorig = np.vstack((exp(-X[:, 0] ** 2 * 10), exp(-X[:, 1] ** 2 * 50), 1.25 * exp(-(X[:, 0] ** 2 + X[:, 1] ** 2) * 5))).max(0)
    Y = np.reshape(Yorig, [N, 1]) + stds * np.random.randn(N, 1)
    return X, Y

def generate_1d_data_simple():
    N = 200
    XX = ((np.random.rand(N,1)-.5)*4)+2;
    YY = np.sin(XX) +np.random.randn(N,1)*0.01
    return XX, YY

def generate_1d_data_nonstationary():
    X = (np.random.rand(N, 1) - 0.5)*4
    Yorig = np.sin(2*X)+2*np.exp(-16*X*X)
    Y = np.reshape(Yorig, [N, 1]) + stds * np.random.randn(N, 1)
    return X, Y

#%%

opt = Options(D)
opt.activ_thresh = 0.4
opt.print_options()

#%%

X, Y = generate_sarcos_data()
Y = np.reshape(Y,(N,1))
print(X.shape, Y.shape)
# print(Y)
model = LGR(opt, D)
debug = False
model.initialize_local_models(X)
initial_local_models = model.get_local_model_activations(X)
nmse = model.run(X, Y, 100, debug)
print("final nmse: {}".format(nmse[-1]))

#%%

Yp = model.predict(X)
final_local_models = model.get_local_model_activations(X)

# print(Yp)
print(final_local_models.shape)
#%%

#matplotlib inline
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(X[:, 0], Y[:, 0], '.g', label='gt')
plt.plot(X[:, 0], Yp[:, 0], '.b', label='predicted')
plt.legend()
plt.subplot(122)
plt.plot(nmse, label='nmse')
plt.legend()
plt.show()

#%%

plt.figure()

plt.plot(X[:, 0], initial_local_models, '.k')
plt.show()

#%%

plt.figure()
plt.plot(X[:, 0], final_local_models, '.k')
plt.show()