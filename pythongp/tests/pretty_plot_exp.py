import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import linear_model
import numpy as np

#data1 = pd.read_csv('/Users/sebastien/Desktop/results/challenge/estimate_param/constant_mean_plugin_matern_3_2_agg_metrics06_19_2019_17_25_35.csv')
#data2 = pd.read_csv('/Users/sebastien/Desktop/results/challenge/estimate_param/three_agg_metrics06_20_2019_01_48_04.csv')
#data3 = pd.read_csv('/Users/sebastien/Desktop/results/challenge/estimate_param/zero_mean_matern_5_2_agg_metrics06_19_2019_17_13_37.csv')

#data = pd.concat((data1, data2, data3), axis = 0)
data = pd.read_csv('/Users/sebastien/Desktop/results/challenge/estimate_param/agg_metrics.csv')

data['lengthscale'] = data['model'].apply(lambda x: 'anisotrope' if 'dimension_specific_lengthscale' in x else 'isotrope')
data['kernel'] = data['model'].apply(lambda x: 'matern52' if 'matern_5_2' in x else 'matern32' if 'matern_3_2' in x else 'rbf' if 'rbf' in x else 'default')
data['mean'] = data['model'].apply(lambda x: 'ord_krig' if 'ord_krig' in x else 'constant' if 'constant' in x else 'zero' if 'zero' in x else 'default')

comparison = 'lengthscale'

#data = data[data['lengthscale'] == 'anisotrope' ]
data = data.groupby([comparison, 'output']).agg({'mse': 'mean', 'is_alpha_credible': 'mean','log_lik': 'sum'}).reset_index()

#s = data.to_string(index = False, float_format = "%.4f")
#s = data.to_string(index = False)
#while '  ' in s:
#    s = s.replace('  ',' ')
#s = s.replace(' ',';')

#print(s)

predictors = ['Orientation1', 'Orientation2', 'Orientation3',
   'Orientation4', 'Orientation5', 'Orientation6', 'Orientation7',
   'Orientation8', 'Orientation9', 'Orientation10', 'Orientation11',
   'Orientation12', 'Orientation13', 'Orientation14', 'Orientation15',
   'Orientation16', 'Orientation17', 'Orientation18', 'Orientation19',
   'Orientation20', 'Orientation21', 'Orientation22']

outputs = ['2F_4N', '1T_5N', 'MAC_1F', 'MAC_2F', 'MAC_1T', 'DELTA_1F_2N',
   'DELTA_1F_3N', 'DELTA_2F_4N', 'DELTA_1T_5N', 'DELTA_TBC_7_8',
   'S11_pli1', 'S11_pli2', 'S11_pli3', 'S11_pli42', 'S11_pli43',
   'S11_pli44', 'Contrainte_max', 'TBC_7_8', 'DELTA_dcalage_70SOL_100',
   'DELTA_dcalage_RLSOL_100', 'dcalage_RLSOL_87', 'dcalage_RLSOL_100',
   'dcalage_70SOL_87', 'dcalage_70SOL_100']

data_set = pd.read_csv('/Users/sebastien/data/data.csv', sep = ';')[outputs]

metric = 'mse'
data_source = 'strat'
type_loo = 'estimate_param'

models = data[comparison].unique().tolist()
outputs = data['output'].unique().tolist()

plt.figure()
plt.semilogy()

type_plot = 'o'
for model in models:
    print(model)
    if model == 'constant_mean_plugin_rbf':
        type_plot = 'x'
    data_model = data[data[comparison] == model]
    data_model = data_model.set_index('output')
    data_model = data_model.loc[outputs]

    #if '_dimension_specific_lengthscale' in model:
    #    model_print = model.replace('_dimension_specific_lengthscale', '_aniso')
    #else:
    #    model_print = model + '_iso'
    model_print = model
    plt.plot(range(len(outputs)), data_model[metric], type_plot, label = model_print)

if metric == 'mse':
    plt.plot(range(len(outputs)), (((data_set[outputs] - data_set[outputs].mean())**2).mean()), label = 'zero predictor')
plt.legend(prop={'size': 6})
#plt.legend(loc=2, prop={'size': 6})
plt.title('{}, {} : {}'.format(data_source,type_loo,metric))
#f = plt.gcf()
#dpi = f.get_dpi()
#h, w = f.get_size_inches()
#f.set_size_inches(h*2, w*2)
#plt.savefig('/Users/sebastien/Desktop/results/pretty_results/'+data_source+'/'+comparison+'/'+type_loo+'/'+type_loo+'_'+metric+'.pdf', format = 'pdf')
plt.show()


full_data_set = pd.read_csv('/Users/sebastien/data/data.csv', sep = ';')

mse = []
for output in ['DELTA_1F_2N']:
    for i in range(full_data_set.shape[0]):
        data_train_loo = full_data_set[full_data_set['Unnamed: 0'] != i]
        data_test_loo = full_data_set[full_data_set['Unnamed: 0'] == i]
        X = data_train_loo[predictors]
        X_test = data_test_loo[predictors]
        y = data_train_loo[output]
        y_test = data_test_loo[output]

        reg = linear_model.LinearRegression().fit(X,y)
        mse.append(((reg.predict(X_test) - y_test)**2).mean())
        #mse.append(((y.mean() - y_test)**2).mean())

print(np.array(mse).mean())

X = full_data_set[predictors]
y = full_data_set[output]
reg = linear_model.LinearRegression().fit(X,y)
((reg.predict(X) - y)**2).mean()