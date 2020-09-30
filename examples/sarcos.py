import numpy as np

import scipy as sc
from scipy import io

from lgr.options import Options
from lgr.batchLGR.lgr import LGR

from sklearn.metrics import mean_squared_error, r2_score

nb_seeds = 5
D_in = 21
nb_outputs = 7

subsample = True
nb_subsample = 100       # if subsample true take every nb_subsample'th training point

# set options
opt = Options(D_in)
opt.activ_thresh = 0.3
opt.max_num_lm = 1000
opt.max_iter = 1000
opt.init_lambda = 0.3   # this lambda value is assuming normalized inputs (roughly between -1 and 1)
opt.do_bwa = False
opt.do_pruning = True
opt.print_options()

# create model
model = LGR(opt, D_in)
debug = False

# load all available data
train_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv.mat')['sarcos_inv']
test_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']

# iterate over seeds
test_mse, test_smse, nb_models = [], [], []
for n in range(nb_seeds):
    print('Training seed', n)
    np.random.seed()

    test_mse_output, test_smse_output, nb_models_output = [], [], []
    # iterate over outputs / joints
    for i in range(nb_outputs):
        print('Training output', i)

        # subsample data
        n_train = train_data.shape[0]
        if subsample:
            train_idx = range(0, n_train, nb_subsample)     # only taking every nb_subsample'th data point
        else:
            train_idx = range(0, n_train, 1)                # all data

        # some rough normalization
        X_train_orig = train_data[train_idx, :D_in]
        Y_train_orig = train_data[train_idx, D_in + i][:, None]
        max_x = np.abs(X_train_orig).max(axis=0)
        # max_y = np.abs(Y_train_orig).max()
        X_train = X_train_orig / max_x
        Y_train = Y_train_orig #/ max_y

        X_test = test_data[:, :D_in] / max_x
        Y_test = test_data[:, D_in+i][:, None] #/ max_y

        # Train
        model.initialize_local_models(X_train)
        initial_local_models = model.get_local_model_activations(X_train)
        nmse = model.run(X_train, Y_train, opt.max_iter, debug)
        # print("TRAIN - NSME: {}".format(nmse[-1]))

        # Test
        _test_pred = model.predict(X_test)

        nb_models_output.append(model.get_local_model_activations(X_test).shape[1])
        test_mse_output.append(mean_squared_error(Y_test, _test_pred))
        test_smse_output.append(1. - r2_score(Y_test, _test_pred, multioutput='variance_weighted'))

        print('OUTPUT - TEST - MSE:', mean_squared_error(Y_test, _test_pred),
              'NMSE:', 1. - r2_score(Y_test, _test_pred, multioutput='uniform_average'),
              'nb_models:', model.get_local_model_activations(X_test).shape[1])

    _nb_models = np.mean(nb_models_output)
    _test_mse = np.sum(test_mse_output)
    _test_smse = np.mean(test_smse_output)

    print('SEED - TEST - MSE:', _test_mse,
              'NMSE:', _test_smse,
              'nb_models:', _nb_models)

    nb_models.append(_nb_models)
    test_mse.append(_test_mse)
    test_smse.append(_test_smse)

mean_nb_models = np.mean(nb_models)
std_nb_models = np.std(nb_models)
mean_mse = np.mean(test_mse)

std_mse = np.std(test_mse)
mean_smse = np.mean(test_smse)
std_smse = np.std(test_smse)

arr = np.array([mean_mse, std_mse,
                mean_smse, std_smse,
                mean_nb_models, std_nb_models])

import pandas as pd

dt = pd.DataFrame(data=arr, index=['mse_avg', 'mse_std',
                                   'smse_avg', 'smse_std',
                                   'models_avg', 'models_std'])
dt.to_csv('results/sarcos_lgp.csv', mode='a', index=True)