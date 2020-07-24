import numpy as np

import scipy as sc
from scipy import io

from lgr.options import Options
from lgr.batchLGR.lgr import LGR

from sklearn.metrics import mean_squared_error, r2_score


opt = Options(21)
opt.activ_thresh = 0.3
opt.max_num_lm = 350
opt.max_iter = 1000
opt.init_lambda = 0.3

opt.print_options()

# load all available data
train_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv.mat')['sarcos_inv']
test_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']

model = LGR(opt, 21)
debug = True

# Train
X_train = train_data[:, :21]
Y_trian = train_data[:, 21][:, None]

model.initialize_local_models(X_train)
initial_local_models = model.get_local_model_activations(X_train)

nmse = model.run(X_train, Y_trian, opt.max_iter, debug)
print("TRAIN - NSME: {}".format(nmse[-1]))

final_local_models = model.get_local_model_activations(X_train)
number_local_models = final_local_models.shape[1]


# Test
X_test = test_data[:, :21]
Y_test = test_data[:, 21][:, None]

Yp = model.predict(X_test)
final_local_models = model.get_local_model_activations(X_test)
_nb_models = final_local_models.shape[1]

_test_mse = mean_squared_error(Y_test, Yp)
_test_smse = 1. - r2_score(Y_test, Yp, multioutput='variance_weighted')

print('FINAL - TEST - MSE:', _test_mse, 'NMSE:', _test_smse, 'nb_models:', _nb_models)
