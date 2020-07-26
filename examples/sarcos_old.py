import numpy as np

from lgr.options import Options
from lgr.batchLGR.lgr import LGR

from sklearn.metrics import mean_squared_error, r2_score

N_train = 3000   # max 44484

D_in = 21

n_seeds = 1
n_sweeps = 1

opt = Options(D_in)
opt.activ_thresh = 0.3
opt.max_num_lm = 1000
opt.max_iter = 3000
opt.init_lambda = 0.3

# opt.alpha_a_0 = 1e-6
# opt.alpha_b_0 = 1e-6
# opt.betaf_a_0 = 1e-6
# opt.betaf_b_0 = 1e-6
opt.alpha_upthresh = 1 + 5e-4 #1.000000001

opt.print_options()

def load_sarcos_data(D_in, N_train):
    import scipy as sc
    from scipy import io

    if int(N_train / 5) > 4449:  # max 4449
        N_test = 4449
    else:
        N_test = int(N_train / 5)

    # load all available data
    _train_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv.mat')['sarcos_inv']
    _test_data = sc.io.loadmat('../datasets/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']

    # shuffle data
    np.random.shuffle(_train_data)
    np.random.shuffle(_test_data)

    data = np.vstack((_train_data, _test_data))

    # scale data with pca
    from sklearn.decomposition import PCA
    input_scaler = PCA(n_components=D_in, whiten=True)
    target_scaler = PCA(n_components=1, whiten=True)

    # # scale data with minmax scaler
    # from sklearn import preprocessing
    # input_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # target_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    input_scaler.fit(data[:, :D_in])
    target_scaler.fit(data[:, D_in:D_in+1])

    train_input = input_scaler.transform(_train_data[:N_train, :D_in])
    # train_target = target_scaler.transform(_train_data[:N_train, D_in:D_in+1])

    # test_input = input_scaler.transform(_test_data[:N_test, :D_in])
    # test_target = target_scaler.transform(_test_data[:N_test, D_in:D_in+1])

    # train_input = data[:N_train, :D_in]
    train_target = data[:N_train, D_in:D_in+1]

    test_input = _test_data[:N_test, :D_in]
    test_target = _test_data[:N_test, D_in:D_in+1]

    return train_input, train_target, test_input, test_target, input_scaler, target_scaler

test_mse, test_smse, nb_models = [], [], []
for i in range(n_seeds):

    print("------------------Seed Nr. " + str(i) + "-----------------------")
    seed = 441
    np.random.seed(seed)

    X_train, Y_train, X_test, Y_test, input_scaler, target_scaler = load_sarcos_data(D_in, N_train)

    model = LGR(opt, D_in)
    debug = False
    model.initialize_local_models(X_train)
    initial_local_models = model.get_local_model_activations(X_train)

    for j in range(n_sweeps):
        print("------------------Sweep Nr. "+str(j)+"-----------------------\n")

        nmse = model.run(X_train, Y_train, opt.max_iter, debug)
        print("FINAL - TRAIN - NSME: {}".format(nmse[-1]))

        final_local_models = model.get_local_model_activations(X_train)
        number_local_models = final_local_models.shape[1]
        print(number_local_models)

    X_test_scaled = input_scaler.transform(X_test)
    # Yp = target_scaler.inverse_transform(model.predict(X_test_scaled))
    final_local_models = model.get_local_model_activations(X_test_scaled)

    Yp = model.predict(X_test_scaled)
    # final_local_models = model.get_local_model_activations(X_test)

    _nb_models = final_local_models.shape[1]

    _test_mse = mean_squared_error(Y_test, Yp)
    _test_smse = 1. - r2_score(Y_test, Yp, multioutput='variance_weighted')

    print('FINAL - TEST - MSE:', _test_mse, 'NMSE:', _test_smse, 'nb_models:', _nb_models)

    test_mse.append(_test_mse)
    test_smse.append(_test_smse)
    nb_models.append(_nb_models)

#%%

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
dt.to_csv('results/sarcos_lgp.csv',mode='a', index=True)



