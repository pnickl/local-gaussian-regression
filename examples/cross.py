import numpy as np

from lgr.options import Options
from lgr.batchLGR.lgr import LGR

from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

#%%

N_train = 2000
N_test = 500
D_out = 1
D_in = 2
seed = 411
np.random.seed(seed)

#%%


def load_sarcos_data(D_out, D_in, N_train, N_test):
    import scipy as sc
    from scipy import io

    x1 = np.arange(-1, 1, 0.001)
    x2 = np.arange(-1, 1, 0.001)
    X1, X2 = np.meshgrid(x1, x2)

    def cross_2d(x1, x2):

        # definition from Vijayakumar et al. Locally Weighted Projection Regression (2000), ICML
        f1 = np.exp(-10 * x1 ** 2)
        f2 = np.exp(-50 * x2 ** 2)
        f3 = 1.25 * np.exp(-5 * (x1 ** 2 + x2 ** 2))
        noise = sc.stats.norm.rvs(np.zeros(len(f1)), 0.01*np.ones(len(f1)), len(f1))
        f_ = np.maximum(f1, f2)
        y = np.maximum(f_, f3) + noise

        # # plot cross function
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.contour3D(X1, X2, y, 100, cmap='binary')

        # train test split
        from sklearn.model_selection import train_test_split
        X = np.vstack((X1, X2)).T
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

        return X_train, X_test, y_train, y_test

    train_input, test_input, train_target, test_target = cross_2d(X1, X2)

    # data = cross_2d(X1, X2)
    # _train_data = data
    # _test_data = data


    #
    # data = np.vstack((_train_data, _test_data))
    #
    # # scale data
    # from sklearn.decomposition import PCA
    # input_scaler = PCA(n_components=21, whiten=True)
    # target_scaler = PCA(n_components=1, whiten=True)
    #
    # input_scaler.fit(data[:, :21])
    # target_scaler.fit(data[:, 21:22])
    #
    # train_data = {'input': input_scaler.transform(_train_data[:N_train, :21]),
    #               'target': target_scaler.transform(_train_data[:N_train, 21:22])}
    #
    # test_data = {'input': input_scaler.transform(_test_data[:N_test, :21]),
    #              'target': target_scaler.transform(_test_data[:N_test, 21:22])}
    #
    # train_input = train_data['input']
    # train_target = train_data['target']
    #
    # test_input = test_data['input']
    # test_target = test_data['target']

    return train_input, train_target, test_input, test_target

#%%

opt = Options(D_out)
opt.activ_thresh = 0.3
opt.max_num_lm = 400
opt.max_iter = 100

# opt.init_eta = 1
# opt.init_lambda = 0.9

opt.print_options()

#%%

X_train, Y_train, X_test, Y_test = load_sarcos_data(D_out, D_in, N_train, N_test)
Y_train, Y_test = np.reshape(Y_train, (N_train, 1)), np.reshape(Y_test, (N_test, 1))

# model = LGR(opt, D_out)
# debug = False
# model.initialize_local_models(X_train)
# initial_local_models = model.get_local_model_activations(X_train)
#
# nmse = model.run(X_train, Y_train, opt.max_iter, debug)
# print("nmse (train): {}".format(nmse[-1]))
#
# #%%
#
# Yp = model.predict(X_test)
# final_local_models = model.get_local_model_activations(X_test)
# print('Number of test data and final local models:', final_local_models.shape)
#
# #%%
# test_mse = mean_squared_error(Y_test, Yp)
# test_smse = 1. - r2_score(Y_test, Yp, multioutput='variance_weighted')
# test_evar = explained_variance_score(Y_test, Yp, multioutput='variance_weighted')
#
# print('FINAL - TEST - MSE:', test_mse, 'SMSE:', test_smse, 'EVAR:', test_evar)
#
# arr = np.array([test_mse, test_smse, test_evar])
# np.savetxt('sarcos_lgp.csv', arr, delimiter=',')
