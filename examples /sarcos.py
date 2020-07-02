import numpy as np
from numpy import exp

from lgr.options import Options
from lgr.batchLGR.lgr import LGR

#%%

N = 1000
D = 21
stds = 0.2
seed = 411
np.random.seed(seed)

#%%

def generate_sarcos_data(D):
    import scipy as sc
    from scipy import io

    # load all available data
    import os
    print(os. getcwd())
    _train_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv.mat')['sarcos_inv']
    _test_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']

    data = np.vstack((_train_data, _test_data))

    train_input = _train_data[:N, :D]
    train_target = _train_data[:N, D]

    test_input = _test_data[:, :D]
    test_target = _test_data[:, D]
    return train_input, train_target, test_input, test_target

#%%

opt = Options(D)
opt.activ_thresh = 0.4
opt.print_options()

#%%

X_train, Y_train, X_test, Y_test = generate_sarcos_data()
Y_train, Y_test = np.reshape(Y_train, (N, 1)), np.reshape(Y_test, (N, 1))
model = LGR(opt, D)
debug = False
model.initialize_local_models(X_train)
initial_local_models = model.get_local_model_activations(X_train)
nmse = model.run(X_train, Y_train, 100, debug)
print("final nmse (train): {}".format(nmse[-1]))

#%%

Yp = model.predict(X_train)
final_local_models = model.get_local_model_activations(X_train)
print('Number of local models:', final_local_models.shape)

