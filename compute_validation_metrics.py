from pythongp.core.wrappers import pgp_init
import pandas as pd
from pythongp.core.params import kernel, mean
import scipy.stats
import numpy as np
import math
from datetime import datetime
from sklearn import linear_model

def update_metrics_with_posterior(y_test, post_mean, post_var, model,
                                  metrics, dimension_specific_lengthscale, alpha, output, row,
                                  c_mean, ls, c_var):

    standard_y_test = (y_test - post_mean) / np.vectorize(math.sqrt)(post_var)

    model_name = model[0]
    if dimension_specific_lengthscale:
        model_name = model_name + '_dimension_specific_lengthscale'

    mse = ((y_test.reshape(-1,1) - post_mean.reshape(-1,1)) ** 2).mean()

    is_alpha_credible = (np.logical_and(scipy.stats.norm.ppf((1 - alpha) / 2) <= standard_y_test,
                                       standard_y_test <= scipy.stats.norm.ppf(1 - (1 - alpha) / 2))).mean()
    log_lik = scipy.stats.norm.logpdf(standard_y_test).sum()

    if max(post_mean.shape) > 1:
        post_mean = [-1]
    else:
        post_mean = post_mean.reshape(1)

    metrics = pd.concat((metrics, pd.DataFrame({'model': [model_name], 'row': [row],
                                                'output': [output],
                                                'mse': [mse],
                                                'is_alpha_credible': [is_alpha_credible],
                                                'log_lik': [log_lik],
                                                'c_mean': c_mean, 'ls_dim_1': ls[0], 'ls_dim_2': ls[1], 'c_var': c_var, 'post_mean' : post_mean})
                         ),
                        ignore_index=True)
    return metrics

def get_proper_estimates(metrics, data, predictors, outputs, model, alpha, in_sample):
    if in_sample:
        indexes = [0]
    else:
        indexes = range(data.shape[0])

    for i in indexes:
        if in_sample:
            data_train_loo = data
            data_test_loo = data
        else:
            data_train_loo = data[data['Unnamed: 0'] != i]
            data_test_loo = data[data['Unnamed: 0'] == i]
        x_train = data_train_loo[predictors].values
        x_test = data_test_loo[predictors].values
        for output in outputs:
            for dimension_specific_lengthscale in [True]:
                y_train = data_train_loo[[output]].values
                y_test = data_test_loo[[output]].values
                trained_model, c_mean, ls, c_var = train_model(model[0], model[1], model[2], x_train, y_train, dimension_specific_lengthscale)

                post_mean, sqrt_post_var = trained_model.predict_multivariate(x_test)
                post_var = sqrt_post_var**2
                #post_mean, post_var = predict_linear_model(trained_model,x_test)

                metrics = update_metrics_with_posterior(y_test, post_mean, post_var, model,
                                  metrics, dimension_specific_lengthscale, alpha, output, i, c_mean, ls, c_var)
    return metrics

def get_fixed_parameters_loo(metrics, data, predictors, outputs, model, alpha):
    for output in outputs:
        for dimension_specific_lengthscale in [True]:
            x_train = data[predictors].values
            y_train = data[[output]].values
            trained_model, c_mean, ls, c_var = train_model(model[0], model[1], model[2], x_train, y_train, dimension_specific_lengthscale)
            post_mean, post_var = predict_gpy_fix_param_loo(trained_model)

            metrics = update_metrics_with_posterior(y_train, post_mean, post_var, model,
                                          metrics, dimension_specific_lengthscale, alpha, output, 1, c_mean, ls, c_var)
    return metrics

def train_model(model_name, mean_function, kernel, x, y, dimension_specific_lengthscale):
    pgp = pgp_init.set_library('GPy')
    pgp.load_multivariate_data(x, y)
    pgp.set_mean(mean_function)
    pgp.set_kernel(kernel, ard = dimension_specific_lengthscale)
    pgp.init_model(noise=0)
    pgp.optimize(param_opt='MLE_with_smart_init', itr=1)

    if 'constant_mean_plugin' in model_name:
        c_mean = pgp.model.constmap.C.values
    else:
        c_mean = [0.0]
    if 'approx_ord_krig' in model_name:
        model_path = pgp.model.sum
    else:
        model_path = pgp.model
    if 'matern_5_2' in model_name:
        ls = model_path.Mat52.lengthscale.values
        c_var = model_path.Mat52.variance.values
    elif 'matern_3_2' in model_name:
        ls = model_path.Mat32.lengthscale.values
        c_var = model_path.Mat32.variance.values
    elif 'rbf' in model_name:
        ls = model_path.rbf.lengthscale.values
        c_var = model_path.rbf.variance.values
    return pgp, c_mean, ls, c_var

def train_linear_model(model_name, mean_function, kernel, x, y, dimension_specific_lengthscale):
    reg = linear_model.LinearRegression().fit(x, y)
    return [reg, ((reg.predict(x) - y)**2).mean()], [0.0], [[0.0], [0.0]], [0.0]

def train_constant_model(model_name, mean_function, kernel, x, y, dimension_specific_lengthscale):
    return [y.mean(), ((y - y.mean())**2).mean()], [0.0], [[0.0], [0.0]], [0.0]

def predict_linear_model(model, x):
    post_mean = model[0].predict(x)
    return post_mean, model[1]*np.ones(x.shape[0])

def predict_constant_model(model, x):
    return model[0]*np.ones(x.shape[0]), model[1]*np.ones(x.shape[0])

def predict_gpy_fix_param_loo(model):
    g = model.model.posterior.woodbury_vector
    c = model.model.posterior.woodbury_inv
    #TODO:() See what happens with normalization
    y = model.model.Y_normalized

    c_diag = np.diag(c)[:, None]

    mu = y - g/c_diag
    var = 1/c_diag

    return mu, var

def get_metrics(file, predictors, outputs, models, reestimate_param, alpha, in_sample):
    data = pd.read_csv(file, sep = ';')

    metrics = pd.DataFrame(columns=['model', 'row', 'output','mse', 'is_alpha_credible', 'log_lik', 'c_mean', 'ls_dim_1', 'ls_dim_2', 'c_var', 'post_mean'])

    cpt = 0
    for model in models:
        if reestimate_param:
            metrics = get_proper_estimates(metrics, data, predictors, outputs, model, alpha, in_sample)
        else:
            metrics = get_fixed_parameters_loo(metrics, data, predictors, outputs, model, alpha)
        cpt = cpt + 1
        print(cpt)
    return metrics

# predictors = ['Orientation1', 'Orientation2', 'Orientation3',
#    'Orientation4', 'Orientation5', 'Orientation6', 'Orientation7',
#    'Orientation8', 'Orientation9', 'Orientation10', 'Orientation11',
#    'Orientation12', 'Orientation13', 'Orientation14', 'Orientation15',
#    'Orientation16', 'Orientation17', 'Orientation18', 'Orientation19',
#    'Orientation20', 'Orientation21', 'Orientation22']
# outputs = ['2F_4N', '1T_5N', 'MAC_1F', 'MAC_2F', 'MAC_1T', 'DELTA_1F_2N',
#    'DELTA_1F_3N', 'DELTA_2F_4N', 'DELTA_1T_5N', 'DELTA_TBC_7_8',
#    'S11_pli1', 'S11_pli2', 'S11_pli3', 'S11_pli42', 'S11_pli43',
#    'S11_pli44', 'Contrainte_max', 'TBC_7_8', 'DELTA_dcalage_70SOL_100',
#    'DELTA_dcalage_RLSOL_100', 'dcalage_RLSOL_87', 'dcalage_RLSOL_100',
#    'dcalage_70SOL_87', 'dcalage_70SOL_100']

predictors = ['x0', 'x1']
outputs = ['f']

models = [
          ('constant_mean_plugin_matern_5_2', mean.Mean().construct('Constant'),
           kernel.Kernel().construct({'Matern': {'lengthscale': 1, 'order': 2.5}})),
          ('approx_ord_krig_matern_5_2', mean.Mean().construct('Zero'),
           kernel.Kernel().construct({'Matern': {'lengthscale': 1, 'order': 2.5}})
           + kernel.Kernel().construct({'Const': {'constant': 1000}})),
          ('zero_mean_matern_3_2', mean.Mean().construct('Zero'),
           kernel.Kernel().construct({'Matern': {'lengthscale': 1, 'order': 1.5}})),
          ('constant_mean_plugin_matern_3_2', mean.Mean().construct('Constant'),
           kernel.Kernel().construct({'Matern': {'lengthscale': 1, 'order': 1.5}})),
          ('approx_ord_krig_matern_3_2', mean.Mean().construct('Zero'),
           kernel.Kernel().construct({'Matern': {'lengthscale': 1, 'order': 1.5}})
           + kernel.Kernel().construct({'Const': {'constant': 1000}})),
          ('zero_mean_rbf', mean.Mean().construct('Zero'),
           kernel.Kernel().construct({'RBF': {'lengthscale': 1}})),
          ('constant_mean_plugin_rbf', mean.Mean().construct('Constant'),
           kernel.Kernel().construct({'RBF': {'lengthscale': 1}})),
          ('approx_ord_krig_rbf', mean.Mean().construct('Zero'),
           kernel.Kernel().construct({'RBF': {'lengthscale': 1}})
           + kernel.Kernel().construct({'Const': {'constant': 1000}})),
           ]

models = [
          ('constant_mean_plugin_matern_5_2', mean.Mean().construct('Constant'),
           kernel.Kernel().construct({'Matern': {'lengthscale': 1, 'order': 2.5}})),
          ]

metrics_loo = get_metrics('/Users/sebastien/data/data_branin.csv', predictors, outputs, models, reestimate_param = True, alpha = 0.95, in_sample = False)
metrics_in_sample = get_metrics('/Users/sebastien/data/data_branin.csv', predictors, outputs,
                                models, reestimate_param = True, alpha = 0.95, in_sample = True)

agg_metrics = metrics_loo.groupby(['model','output']).agg({'mse': ['mean', 'var'], 'c_mean' : ['mean','var'], 'c_var' : ['mean','var'],
                                                       'ls_dim_1' : ['mean','var'], 'ls_dim_2' : ['mean','var']})
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(agg_metrics[['mse', 'c_mean', 'c_var', 'ls_dim_1', 'ls_dim_2']])
#
#
# metrics_loo['kernel'] = metrics_loo['model'].apply(lambda x: 'matern52' if 'matern_5_2' in x else 'matern32' if 'matern_3_2' in x else 'rbf' if 'rbf' in x else 'default')
# metrics_in_sample['mean'] = metrics_in_sample['model'].apply(lambda x: 'ord_krig' if 'ord_krig' in x else 'constant' if 'constant' in x else 'zero' if 'zero' in x else 'default')
# #
# metrics_dummy['kernel'] = metrics_dummy['model'].apply(lambda x: 'matern52' if 'matern_5_2' in x else 'matern32' if 'matern_3_2' in x else 'rbf' if 'rbf' in x else 'default')
# metrics_dummy['mean'] = metrics_dummy['model'].apply(lambda x: 'ord_krig' if 'ord_krig' in x else 'constant' if 'constant' in x else 'zero' if 'zero' in x else 'default')
#
#
#
# import matplotlib.pyplot as plt
#
# variable = 'c_mean'
#
# #plt.locator_params(axis='x', nbins=4)
# plt.figure()
#
# i = 1
#
# plt.figure()
# plt.suptitle("{}".format(variable))
# i = 1
# for k in metrics['kernel'].unique():
#     plt.subplot(1, 3, i)
#     plt.title(k)
#     #plt.semilogx()
#     #fig1, ax1 = plt.subplots()
#     #ax1.set_xscale('log')
#
#     print(metrics[(metrics['mean'] == 'constant') & (metrics['kernel'] == k)][variable])
#     plt.hist(metrics[(metrics['mean'] == 'constant') & (metrics['kernel'] == k)][variable], label = 'loo estimates')
#     plt.vlines(metrics_dummy[(metrics_dummy['mean'] == 'constant') & (metrics_dummy['kernel'] == k)][variable], ymin = 0, ymax = 20, label = 'full data estimate')
#     plt.legend()
#     i = i + 1
#
# plt.show()
#
# import matplotlib.pyplot as plt
#
# variable = 'ls_dim_2'
#
# for k in metrics['kernel'].unique():
#     plt.figure()
#     plt.subplots(figsize=(15, 10))
#     plt.suptitle("{} : {}".format(k, variable))
#     i = 1
#     for m in metrics['mean'].unique():
#         plt.subplot(1, 3, i)
#         plt.title(m)
#         #plt.semilogx()
#         #fig1, ax1 = plt.subplots()
#         #ax1.set_xscale('log')
#
#         print(metrics[(metrics['mean'] == m) & (metrics['kernel'] == k)][variable])
#         plt.vlines(metrics_dummy[(metrics_dummy['mean'] == m) & (metrics_dummy['kernel'] == k)][variable].apply(math.log10),
#                    ymin=0, ymax=20, label='full data estimate')
#         plt.hist(metrics[(metrics['mean'] == m) & (metrics['kernel'] == k)][variable].apply(math.log10),  label = 'full data estimate')
#         i = i + 1
#     #plt.show()
#     plt.savefig("/Users/sebastien/Desktop/presentation_resultats_bect_vazquez/images_branin/{}_{}.png".format(k, variable))

#agg_metrics.to_csv('/Users/sebastien/Desktop/proper_loo'+datetime.now().strftime('%m_%d_%Y_%H_%M_%S')+'.csv')

# key = (metrics['model'] == 'constant_mean_plugin_matern_5_2_dimension_specific_lengthscale')
# key = key | (metrics['model'] == 'constant_mean_plugin_rbf_dimension_specific_lengthscale')
# test = metrics[key][['model', 'row', 'post_mean']]
#
# test['model'] = test['model'].apply(lambda x : x.replace('constant_mean_plugin_','').replace('_dimension_specific_lengthscale', ''))
#
# data_train = pd.read_csv('/Users/sebastien/data/data_branin.csv', sep = ';')

#import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix

#plt.figure()
#plt.plot(data_train['f'], data_test['post_mean'] , 'o')
#plt.show()
#
# to_plot = test.pivot(columns = 'model', values = 'post_mean', index = 'row')
#
# to_plot = pd.concat((to_plot, data_train['f']), axis = 1)
#
# #scatter_matrix(to_plot, alpha = 1)
#
# ###########################
#
# key = (metrics['model'] == 'constant_mean_plugin_matern_5_2_dimension_specific_lengthscale')
# test_tmp = metrics[key]
#
# test_tmp = test_tmp.reset_index(drop = True)
#
# arg_max = (test_tmp['mse'].values).argmax()