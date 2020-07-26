import numpy as np
import numpy.random as npr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from lwpr import *
# load all available data
train_input = np.load('data/wam/wam_inv_train.npz')['input']
train_target = np.load('data/wam/wam_inv_train.npz')['target']
test_input = np.load('data/wam/wam_inv_test.npz')['input']
test_target = np.load('data/wam/wam_inv_test.npz')['target']
np.random.seed(1337)
nb_seeds = 5
nb_sweeps = 2
nb_outputs = train_target.shape[-1]
# train
test_mse, test_smse, nb_models = [], [], []
for n in range(nb_seeds):
    print('Training seed', n)
    _test_mu = np.zeros((len(test_input), nb_outputs))
    _nb_models = np.zeros((nb_outputs, ))
    for i in range(nb_outputs):
        print('Training output', i)
        model = LWPR(18, 1)
        model.norm_in = np.var(train_input, axis=0)
        model.init_D = 0.25 * np.diag(1. / np.var(train_input, axis=0))
        model.update_D = True
        model.init_alpha = 0.25 * np.eye(18)
        model.meta = True
        model.penalty = 1e-6
        model.diag_only = True
        for k in range(nb_sweeps):
            _input, _target = shuffle(train_input, train_target)
            for j in range(len(train_input)):
                model.update(_input[j, :], _target[j, i][None])
        _nb_models[i] = model.num_rfs[0]
        for j in range(len(test_input)):
            _test_mu[j, i] = model.predict(test_input[j, :])
    _test_mse = mean_squared_error(test_target, _test_mu)
    _test_smse = 1. - r2_score(test_target, _test_mu)
    print('TRAIN - MSE:', _test_mse,
          'SMSE:', _test_smse,
          'Models:', _nb_models.sum())
    test_mse.append(_test_mse)
    test_smse.append(_test_smse)
    nb_models.append(_nb_models.sum())
mean_mse = np.mean(test_mse)
std_mse = np.std(test_mse)
mean_smse = np.mean(test_smse)
std_smse = np.std(test_smse)
mean_nb_models = np.mean(nb_models)
std_nb_models = np.std(nb_models)
arr = np.array([mean_mse, std_mse,
                mean_smse, std_smse,
                mean_nb_models, std_nb_models])
import pandas as pd
dt = pd.DataFrame(data=arr, index=['mse_avg', 'mse_std',
                                   'smse_avg', 'smse_std',
                                   'models_avg', 'models_std'])
dt.to_csv('wam_lwpr.csv', mode='a', index=True)