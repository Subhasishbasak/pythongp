import matplotlib.pyplot as plt
import pandas as pd

#data1 = pd.read_csv('/Users/sebastien/Desktop/results/challenge/estimate_param/constant_mean_plugin_matern_3_2_agg_metrics06_19_2019_17_25_35.csv')
#data2 = pd.read_csv('/Users/sebastien/Desktop/results/challenge/estimate_param/three_agg_metrics06_20_2019_01_48_04.csv')
#data3 = pd.read_csv('/Users/sebastien/Desktop/results/challenge/estimate_param/zero_mean_matern_5_2_agg_metrics06_19_2019_17_13_37.csv')

#data = pd.concat((data1, data2, data3), axis = 0)
data = pd.read_csv('/Users/sebastien/Desktop/results/g_10/estimate_param/estimate_param_g10_06_20_2019_10_47_58.csv')

data_set = pd.read_csv('/Users/sebastien/data/data_g10.csv', sep = ';')

metric = 'mse'

models = data['model'].unique().tolist()
outputs = data['output'].unique().tolist()

plt.figure()

type_plot = 'o'
for model in models:
    print(model)
    if model == 'constant_mean_plugin_rbf':
        type_plot = 'x'
    data_model = data[data['model'] == model]
    data_model = data_model.set_index('output')
    data_model = data_model.loc[outputs]

    if '_dimension_specific_lengthscale' in model:
        model_print = model.replace('_dimension_specific_lengthscale', '_aniso')
    else:
        model_print = model + '_iso'
    plt.plot(range(len(outputs)), data_model[metric], type_plot, label = model_print)

if metric == 'mse':
    plt.plot(range(len(outputs)), ((data_set[outputs] - data_set[outputs].mean())**2).mean(), label = 'zero predictor')
plt.legend(prop={'size': 6})
#plt.legend(loc=2, prop={'size': 6})
plt.title('g10, proper loo : {}'.format(metric))
plt.show()
