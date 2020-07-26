import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import numpy.random as npr

import scipy as sc
from scipy import io

from lgr.options import Options
from lgr.batchLGR.lgr import LGR

from sklearn.metrics import mean_squared_error, r2_score

import pathos
from pathos.pools import ProcessPool as Pool
nb_cores = pathos.multiprocessing.cpu_count()


np.random.seed(1337)


def _job(kwargs):

    idx = kwargs.pop('idx')
    X_train = kwargs.pop('input')
    Y_train = kwargs.pop('target')[:, [idx]]

    seed = kwargs.pop('seed')

    # set random seed
    np.random.seed(seed)

    # set options
    opt = Options(21)
    opt.activ_thresh = 0.3
    opt.max_num_lm = 1000
    opt.max_iter = 1000
    opt.init_lambda = 0.3
    opt.do_bwa = False
    opt.do_pruning = True
    # opt.print_options()

    # create model
    model = LGR(opt, 21)

    # Train
    model.initialize_local_models(X_train)
    model.run(X_train, Y_train, opt.max_iter, debug=False)

    return model


def _parallelize(nb_seeds=5, nb_outputs=7, **kwargs):
    kwargs_list = []

    for n in range(nb_seeds):
        seed = npr.randint(1337, 6174)
        for i in range(nb_outputs):
            kwargs['seed'] = seed
            kwargs['idx'] = i
            kwargs_list.append(kwargs.copy())

    with Pool(processes=min(nb_seeds * nb_outputs, nb_cores)) as p:
        res = p.map(_job, kwargs_list)

    return res


# load all available data
_train_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv.mat')['sarcos_inv']
_test_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']

nb_inputs = 21
nb_outputs = 7
nb_seeds = 5

_train_input = _train_data[:, :nb_inputs]
_test_input = _test_data[:, :nb_inputs]

input_data = np.vstack((_train_input, _test_input))
x_max = np.max(np.abs(input_data), axis=0)

train_input = _train_input / x_max
test_input = _test_input / x_max

train_target = _train_data[:, nb_inputs:]
test_target = _test_data[:, nb_inputs:]

models = _parallelize(nb_seeds=nb_seeds, nb_outputs=nb_outputs,
                      input=train_input, target=train_target)

nb_models, test_mse, test_smse = [], [], []

for n in range(nb_seeds):
    _test_mu = np.zeros((len(test_input), nb_outputs))
    _nb_models = np.zeros((nb_outputs, ))

    for i in range(nb_outputs):
        _model = models[n * nb_outputs + i]

        _test_mu[:, i] = _model.predict(test_input).squeeze()
        _nb_models[i] = _model.get_local_model_activations(train_input).shape[1]

    _test_mse = mean_squared_error(test_target, _test_mu)
    _test_smse = 1. - r2_score(test_target, _test_mu)

    print('TEST - MSE:', _test_mse,
          'NMSE:', _test_smse,
          'nb_models:', _nb_models.sum())

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
dt.to_csv('sarcos_lgp.csv', mode='a', index=True)
