import numpy as np
from numpy import exp

from lgr.options import Options
from lgr.batchLGR.lgr import LGR

from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

#%%

N_train = 500
N_test = 100
D = 21

seed = 411
np.random.seed(seed)

#%%

def generate_sarcos_data(D, N_train, N_test):
    import scipy as sc
    from scipy import io

    # load all available data
    import os

    _train_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv.mat')['sarcos_inv']
    _test_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']

    data = np.vstack((_train_data, _test_data))

    # scale data
    from sklearn.decomposition import PCA
    input_scaler = PCA(n_components=21, whiten=True)
    target_scaler = PCA(n_components=7, whiten=True)

    input_scaler.fit(data[:, :21])
    target_scaler.fit(data[:, 21:28])

    train_data = {'input': input_scaler.transform(_train_data[:N_train, :21]),
                  'target': target_scaler.transform(_train_data[:N_train, 21:28])}

    test_data = {'input': input_scaler.transform(_test_data[:N_test, :21]),
                 'target': target_scaler.transform(_test_data[:N_test, 21:28])}

    # train_input = train_data[:N_train, :D]
    # train_target = train_data[:N_train, D]
    #
    # test_input = test_data[:N_test, :D]
    # test_target = test_data[:N_test, D]

    train_input = train_data['input']
    train_target = train_data['target']

    test_input = test_data['input']
    test_target = test_data['target']

    return train_input, train_target, test_input, test_target

#%%

opt = Options(D)
opt.activ_thresh = 0.3
opt.max_num_lm = 1000
opt.max_iter = 500
opt.print_options()

#%%

X_train, Y_train, X_test, Y_test = generate_sarcos_data(D, N_train, N_test)
Y_train, Y_test = np.reshape(Y_train, (N_train, 1)), np.reshape(Y_test, (N_test, 1))

model = LGR(opt, D)
debug = False
model.initialize_local_models(X_train)
initial_local_models = model.get_local_model_activations(X_train)

nmse = model.run(X_train, Y_train, opt.max_iter, debug)
print("final nmse (train): {}".format(nmse[-1]))

#%%

Yp = model.predict(X_test)
final_local_models = model.get_local_model_activations(X_test)
print('Number of data and local models:', final_local_models.shape)

#%%
test_mse = mean_squared_error(Y_train, Yp)
test_smse = 1. - r2_score(Y_train, Yp, multioutput='variance_weighted')
test_evar = explained_variance_score(Y_train, Yp, multioutput='variance_weighted')

print('FINAL - TEST - MSE:', test_mse, 'SMSE:', test_smse, 'EVAR:', test_evar)

arr = np.array([test_mse, test_smse, test_evar])
np.savetxt('sarcos_lgp.csv', arr, delimiter=',')
