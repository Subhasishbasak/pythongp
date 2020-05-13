# coding: utf-8
import time
import math
import scipy
import numpy as np
import pandas as pd
import matplotlib.mlab as ml
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from pythongp.core.params import kernel, mean
from pythongp.test_functions import test_functions

def make_data(input_dim, test_func, x_1_bounds, *arg):
        '''
        This function generates training observation of size "n" randomly from a
        Uniform (x_min, x_max) distribution & calls it "x_train".
        It evaluate the test function on "x_train" & calls it "z_train"
        Stores the training data in a dataframe and generates a plot
        '''
        np.random.seed(0)
        n = 100 # number of training data points
        m = 1000 # number of test data points
        
        if input_dim == 1:
            x_min = x_1_bounds[0]
            x_max = x_1_bounds[1]
            x_train = list (np.random.uniform (low = x_min, high = x_max, size = n))
            x_train = list (np.sort(x_train))
            z_train = list (map (test_func, x_train))
            x_test = np.linspace(x_min, x_max, m)
            z_test = list(map(test_func, x_test))
            
            train_data_frame = pd.DataFrame ()
            train_data_frame["x"] = x_train
            train_data_frame["z_train"] = z_train
            
            test_data_frame = pd.DataFrame ()
            test_data_frame["x"] = x_test
            test_data_frame["z_test"] = z_test
            
            # Plotting the training data
    
            plt.figure()
            plt.plot (train_data_frame['x'], train_data_frame['z_train'])
            plt.plot (train_data_frame['x'], train_data_frame['z_train'], 'r.', markersize=10, label=u'Observations')
            plt.xlabel('$x$')
            plt.ylabel('$z = f(x)$')
            plt.legend(loc='upper left')
            plt.show()
            print("Data loading done")
            
            return train_data_frame, test_data_frame


        else:
            bounds = {}
            bounds[0] = [x_1_bounds[0], x_1_bounds[1]]
            for i in range(1, input_dim) :
                bounds[i] = arg[i-1]
            x_train_dict = {}
            x_test_dict = {}
            for i in range(0,input_dim):
                x_train_dict[i] = np.random.uniform(bounds[i][0],bounds[i][1],n)    
            x_train = np.array(list(zip(*x_train_dict.values())))
            z_train = np.array(list (map (test_func, x_train[:,0], x_train[:,1])))
            for i in range(0,input_dim):
                x_test_dict[i] = np.random.uniform(bounds[i][0],bounds[i][1],m)
            x_test = np.array(list(zip(*x_test_dict.values())))
            z_test = np.array(list (map (test_func, x_test[:,0], x_test[:,1])))
                
            return x_train,z_train,x_test,z_test


def plot (test_data_frame, train_data_frame, postmean, postvar):

    '''
    This function creates a plot of the training, test & predicted data
    '''

    if postmean is None or postvar is None:
        return


    plt.figure()

    plt.plot(test_data_frame['x'], test_data_frame['z_test'], 'r:', label=r'Test function')

    plt.plot(train_data_frame['x'], train_data_frame['z_train'], 'r.', markersize=10, label=u'Observations')

    plt.plot(test_data_frame['x'], postmean, 'b-', label=u'Prediction')

    plt.fill(np.concatenate([test_data_frame['x'], test_data_frame['x'][::-1]]),
         np.concatenate([postmean - 1.9600 * postvar,
                        (postmean + 1.9600 * postvar)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')

    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    plt.show()


def accuracy (test_data_frame, postmean, *postvar):

    z_test = test_data_frame['z_test']
    if postmean is None :
        return
    emrmse = np.sqrt(np.mean(np.square(np.array(z_test)-np.array(postmean))))
    pmrmse = np.sqrt(np.mean(np.square(np.array(postvar))))
    print ("\nThe R-square value is : ",r2_score(list(z_test), list(postmean)))    
    print ("Corr. coeff. b/w z_test & z_pred : ",scipy.stats.pearsonr(list(z_test), list(postmean))[0])
    print("EMRMSE",emrmse)
    print("PMRMSE",pmrmse)
    
    
def emrmse(test_data_frame, postmean, postvar):
    z_test = test_data_frame['z_test']
    if postmean is None :
        return
    emrmse = np.sqrt(np.mean(np.square(np.array(z_test)-np.array(postmean))))
    return emrmse



def pmrmse(postvar):
    pmrmse = np.sqrt(np.mean(np.square(np.array(postvar))))
    return pmrmse
    
        
def contour(x_test, z_test, z_postmean, test_func):

    x = x_test[:,0]
    y = x_test[:,1]

    delta = 0.1         # grid density
    X, Y = np.meshgrid(np.arange(-5,10, delta), np.arange(0,15, delta))
    Z = np.vectorize(test_func)(X,Y)
    
    plt.figure()
    cp = plt.contourf(X, Y, Z )
    
    plt.colorbar(cp)
    plt.scatter(x, y, marker = 'o', c = z_postmean, s = 5, zorder = 10)
    plt.xlabel('x ')
    plt.ylabel('y ')
    plt.show()
    
    

def test01(pgp):
    '''
    test01
    Author: S.B
    Description: The following code implements Gaussian Process regression using
    a perticular python libraries for a given test function
    '''
    ############################ defining test function #############################

    test_func, x_min, x_max = test_functions.f01, test_functions.func_domain[1][0], test_functions.func_domain[1][1]

    train_data, test_data = make_data(1,test_func, [x_min, x_max])

    ############################## Initializing the test ############################


    ################################## load data ####################################

    pgp.load_data(train_data)

    ########################### Constructing the Kernel #############################

    # with pre specified inputs

    
    kernel_dict_input = {}
    #kernel_dict_input['Matern'] = {'lengthscale': 1, 'order': 1.5, 'lengthscale_bounds': '(1e-5, 1e5)'}
    kernel_dict_input['RBF'] = {'lengthscale': 1, 'lengthscale_bounds': '(1e-5, 1e5)'}

    k = kernel.Kernel()
    k.construct(kernel_dict_input)
    
    pgp.set_kernel (k)
    '''


    # with user inputs

    k1 = kernel.Kernel()
    k1.construct()

    #k2 = kernel.Kernel()
    #k2.construct()

    #k3 = kernel.Kernel()
    #k3.construct()

    #k = kernel.CompoundKernel('+',kernel.CompoundKernel('*',kernel.CompoundKernel(o = k1.show()),kernel.CompoundKernel(o = k2.show())),kernel.CompoundKernel(o = k3.show()))
    #k = (k1 * k2) + k3

    pgp.set_kernel (k1)
    '''

    ######################## Specifying the Mean Function ###########################

    # with pre specified inputs
    
    m = mean.Mean()
    m.construct('Zero')
    pgp.set_mean (m)
    
    

    # with user inputs
    '''
    m = mean.Mean()
    m.construct()
    pgp.set_mean (m)
    '''
    ##################### Construction of the regression model ######################

    pgp.init_model(noise=0.001)
    pgp.optimize(param_opt='MLE', itr=10)
    
    ############################## Making predictions ###############################

    z_postmean, z_postvar = pgp.predict (test_data)

    ########################## Plotting the predictions #############################

    plot (test_data, train_data, z_postmean, z_postvar)

    ############################# Computing accuracy ################################

    accuracy(test_data, z_postmean, z_postvar)


def test02(pgp):
    '''
    test02
    Author: S.B
    Description: The following code implements Gaussian Process regression and
    compares the prediction accuracies of different python libraries for a given
    test function
    '''

    # defining test function

    test_func, x_min, x_max = test_functions.f03, test_functions.func_domain[3][0], test_functions.func_domain[3][1]

    # generate data

    train_data, test_data = make_data(1, test_func, [x_min, x_max])

    '''
    Afterwards this test works with the same loaded data for each library
    '''

    # Reshaping the data according to library configuration

    pgp.load_data(train_data)

    # Constructing the Kernel

    kernel_dict_input = {}
    #kernel_dict_input['Matern'] = {'lengthscale': 1, 'order': 1.5, 'lengthscale_bounds': '(1e-5, 1e5)'}
    kernel_dict_input['RBF'] = {'lengthscale': 1, 'lengthscale_bounds': '(1e-5, 1e5)', 'scale': 1}
    #kernel_dict_input['RatQd'] = {'lengthscale': 1 , 'power': 1.5, 'lengthscale_bounds': '(1e-5, 1e5)'}
    k = kernel.Kernel()
    k.construct(kernel_dict_input)

    pgp.set_kernel (k)

    # Specifying the Mean Function

    m = mean.Mean()
    m.construct('Zero')
    pgp.set_mean (m)

    # Construction of the regression model

    pgp.init_model(noise=0.0001)
    pgp.optimize(param_opt='MLE', itr=10)

    # Making predictions

    z_postmean, z_postvar = pgp.predict(test_data)
    

    # Plotting the predictions

    plot(test_data, train_data, z_postmean, z_postvar)

    # Computing accuracy

    accuracy(test_data, z_postmean, z_postvar)


def test_zero_accumulation(pgp, n_start, n_stop, n_test, eps, noise, n_restart):
    '''
    test_zero_accumulation
    Author: S.B
    This test compare the issue with an accumulation point for several toolboxes.
    '''
    # defining test function

    test_func = test_functions.f08

    '''
    Afterwards this test works with the same loaded data for each library
    '''

    x_train = [((-1) ** n) / n for n in range(1, n_start)]
    y_train = [test_func(x) for x in x_train]

    x_test = np.linspace(-1,20, n_test)
    y_test = [test_func(x) for x in x_test]

    n_zero_var = []
    krig_error = []
    one_loess = []
    two_loess = []
    x_new = x_train[-1]
    y_new = y_train[-1]
    times = []
    non_zero_var_means = []

    t = time.time()

    for k in range(n_start, n_stop + 1):
        print("\n")
        print("Number points : {}".format(k))
        old_t = t
        t = time.time()
        print("Time elapsed : {}".format(t - old_t))
        times.append(t - old_t)

        old_x_new = x_new
        x_new = ((-1) ** k) / k
        x_train.append(x_new)

        old_y_new = y_new
        y_new = test_func(x_new)
        y_train.append(y_new)

        # Reshaping the data according to library configuration
        # print(len(x_train))
        # print(len(y_train))
        train_data = pd.DataFrame({'x': x_train, 'z_train': y_train, 'z_test' : y_train})
        test_data = pd.DataFrame({'x': x_test, 'z_test': y_test})

        pgp.load_data(train_data)

        # Specifying the Mean Function

        m = mean.Mean()
        m.construct('Zero')
        pgp.set_mean (m)


        # Constructing the Kernel

        kernel_dict_input = {}
        kernel_dict_input['Matern'] = {'lengthscale': 1, 'order': 2.5}
        #kernel_dict_input['RBF'] = {'lengthscale': 1, 'lengthscale_bounds': '(1e-5, 1e5)'}

        k1 = kernel.Kernel()
        k1.construct(kernel_dict_input)

        k2 = kernel.Kernel()
        k2.construct({'Const':{'constant' : 1000}})
        pgp.set_kernel (k1 + k2)
        #pgp.set_kernel(k1)

        # Construction of the regression model
        try:
            pgp.init_model(noise = noise)
            pgp.optimize(param_opt = 'MLE', itr = n_restart)
            print('pgp model : {}'.format(pgp.model))
        except np.linalg.LinAlgError as err:
            print("Error : {}".format(err))
            n_zero_var.append(0)
            non_zero_var_means.append(0)
            krig_error.append(1)
            one_loess.append((y_new) ** 2)
            two_loess.append((y_new - x_new * (old_y_new - y_new) / (old_x_new - x_new)) ** 2)
            continue

        # Making predictions

        z_postmean_test, sqrt_z_postvar_test = pgp.predict(test_data)
        z_postvar_test = sqrt_z_postvar_test**2

        n_zero_var.append((z_postvar_test <= eps).sum())
        non_zero_var_means.append((z_postvar_test[z_postvar_test > eps]).mean())

        z_postmean_zero, _ = pgp.predict(pd.DataFrame({'x': [0.0], 'z_test': [0.0]}))
        krig_error.append((z_postmean_zero) ** 2)
        one_loess.append((y_new) ** 2)
        two_loess.append((y_new - x_new * (old_y_new - y_new) / (old_x_new - x_new)) ** 2)
        # Plotting the predictions

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(n_start, n_stop + 1), n_zero_var, 'k:', label=r'Number of null variances')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.semilogy(range(n_start, n_stop + 1), krig_error, 'k:', label=r'Error in zero')
    plt.semilogy(range(n_start, n_stop + 1), one_loess, 'r-', label=r'1NN in zero')
    plt.semilogy(range(n_start, n_stop + 1), two_loess, 'b-', label=r'2NN regression in zero')
    plt.legend()

    plot(test_data, train_data, z_postmean_test, z_postvar_test)
    # plt.subplot(2,2,3)
    # plt.plot(range(n_start, n_stop + 1), times, 'k:', label=r'Execution time')
    # plt.legend()

    # plt.subplot(2,2,4)
    # plt.plot(range(n_start, n_stop + 1), non_zero_var_means, 'k:', label=r'Non zero variance mean')
    # plt.legend()

def test_nugget(pgp, n):
    '''
    test_zero_accumulation
    Author: S.B
    This test compare the issue with an accumulation point for several toolboxes.
    '''
    # defining test function

    test_func = test_functions.f08

    '''
    Afterwards this test works with the same loaded data for each library
    '''

    x_train = [((-1) ** n) / n for n in range(1, n)]
    y_train = [test_func(x) for x in x_train]

    x_new = x_train[-1]

    # Reshaping the data according to library configuration

    # print(len(x_train))
    # print(len(y_train))
    train_data = pd.DataFrame({'x': x_train, 'z_train': y_train, 'z_test' : y_train})

    pgp.load_data(train_data)

    # Specifying the Mean Function

    m = mean.Mean()
    m.construct('Zero')
    pgp.set_mean(m)

    # Constructing the Kernel

    k = kernel.Kernel()

    kernel_dict_input = {}
    kernel_dict_input['Matern'] = {'lengthscale': 1, 'order': 2.5, 'lengthscale_bounds': '(1e-5, 1e5)'}
    k.construct(kernel_dict_input)
    pgp.set_kernel(k)

    # Construction of the regression model
    pgp.init_model(noise=0)
    pgp.optimize(param_opt='MLE', itr=10)
    #print('pgp model : {}'.format(pgp.model))
    print('Hello world')
    # Making predictions
    _, sqrt_z_postvar_zero = pgp.predict(pd.DataFrame({'x': [0.0], 'z_test': [0.0]}))

    z_postvar_zero = sqrt_z_postvar_zero**2
    var_zero = pgp.evaluate_ker(np.array([[0]]), np.array([[0]]))
    var_last_point = pgp.evaluate_ker(np.array([[x_new]]), np.array([[x_new]]))
    cov_zero_last_point = pgp.evaluate_ker(np.array([[x_new]]), np.array([[0]]))

    upper_bound = var_zero  + var_last_point - 2 * cov_zero_last_point
    
    print('post_var' , z_postvar_zero[0])
    print('upper_bound' , upper_bound[0,0])
    print(z_postvar_zero[0] <= upper_bound[0,0] )


def test_multivar(pgp):
    '''
    test_multivar
    Author: S.B
    Description: The following code implements Gaussian Process regression and
    compares the prediction accuracies of different python libraries for a given
    test function
    '''

    # defining test function
    

    test_func, x_1_min, x_1_max, x_2_min, x_2_max  = test_functions.branin, test_functions.func_domain_2d['branin'][0][0], test_functions.func_domain_2d['branin'][0][1],test_functions.func_domain_2d['branin'][1][0], test_functions.func_domain_2d['branin'][1][1]

    # generate data
    
    input_dim = 2
    x_train, z_train, x_test, z_test = make_data(input_dim, test_func, [x_1_min, x_1_max], [x_2_min, x_2_max])

    '''
    Afterwards this test works with the same loaded data for each library
    '''

    # Reshaping the data according to library configuration

    pgp.load_mult_data(x_train, z_train)

    # Constructing the Kernel

    kernel_dict_input = {}
    #kernel_dict_input1 = {}
    #kernel_dict_input['Matern'] = {'lengthscale': [2,3], 'order': 1.5, 'lengthscale_bounds': '(1e-5, 1e5)'}
    kernel_dict_input['RBF'] = {'lengthscale': [1,1] , 'lengthscale_bounds': '(1e-5, 1e5)', 'scale': 1}
    #kernel_dict_input['RatQd'] = {'lengthscale': 1 , 'power': 1.5, 'lengthscale_bounds': '(1e-5, 1e5)'}
    #kernel_dict_input1['Const'] = {'constant': 1}
    
    k = kernel.Kernel()
    k.construct(kernel_dict_input)
    
    #k1 = kernel.Kernel()
    #k1.construct(kernel_dict_input1)
    
    pgp.set_kernel (k, ard = True)
    '''
    Set ard = True for different lenghscale for different dimensions
    '''
    # Specifying the Mean Function

    m = mean.Mean()
    m.construct('Zero')
    pgp.set_mean (m)

    # Construction of the regression model

    pgp.init_model(noise=0.0001)
    pgp.optimize(param_opt='MLE', itr=10)
    
    # Making predictions

    z_postmean, z_postvar = pgp.predict_mult(x_test)

    # Plotting the predictions

    #plot(test_data, train_data, z_postmean, z_postvar)

    # Computing accuracy
    
    test_data = pd.DataFrame()
    test_data["z_test"] = z_test
    accuracy(test_data, z_postmean, z_postvar)
    
    # Contour plots
    
    contour(x_test, z_test, z_postmean, test_func)


def test_simulate(pgp, n):
    '''
    test_simulate
    Author: S.B
    Description: The following code implements Gaussian Process regression and
    generates boxplots of accuracy measures. (in multivariate setup)
    '''
    
    pmrmse_dict = {}
    emrmse_dict = {}
    time_dict = {}
    for i in range(1,n+1):
        print('set seed : ',i)
        # defining the random surface
        '''
        def test_bed_1(x, y):
            def generate_coeff():
                np.random.seed(i)
                R1 = float(np.random.uniform (low = 0, high = 0.5, size = 1))
                S1 = float(np.random.uniform (low = 0, high = 0.5, size = 1))
                R2 = float(np.random.uniform (low = 0.5, high = 1, size = 1))
                S2 = float(np.random.uniform (low = 0.5, high = 1, size = 1))
                A = float(np.random.uniform (low = 0, high = 0.05, size = 1))
                Z = float(np.random.uniform (low = 0.25, high = 1, size = 1))
                return R1,S1,R2,S2,A,Z
                
            R1,S1,R2,S2,A,Z = generate_coeff()
            return (x**3/3-(R1+S1)*x**2/2+(R1*S1)*x+y**3/3-(R2+S2)*y**2/2+(R2*S2)*y+A*math.sin(2*math.pi*x*y/Z))
        
        def test_bed_2(x, y):
            def generate_coeff_2():
                    np.random.seed(1)
                    A1 = float(np.random.uniform (low = 20, high = 35, size = 1))
                    A2 = float(np.random.uniform (low = 20, high = 35, size = 1))
                    B1 = float(np.random.uniform (low = 0.5, high = 0.9, size = 1))
                    B2 = float(np.random.uniform (low = 0.5, high = 0.9, size = 1))
                    C = 10
                    return A1,A2,B1,B2,C
                
            A1,A2,B1,B2,C = generate_coeff_2()
            return (C*(math.sin(A1*(abs(x-0.5)-B1)**4)*math.cos(2*abs(x-0.5)-B1)+((abs(x-0.5)-B1)/2))*(math.sin(A2*(abs(y-0.5)-B2)**4)*math.cos(2*abs(y-0.5)-B2)+((abs(y-0.5)-B2)/2)))
        '''
        def test_bed_2(*x):
            size = len(x)
            def generate_coeff_2():
                    np.random.seed(i)
                    A = (np.random.uniform (low = 20, high = 35, size = size))
                    B = (np.random.uniform (low = 0.5, high = 0.9, size = size))
                    C = 10
                    return A,B,C
                
            A,B,C = generate_coeff_2()
            ans = 1
            for j in range(0,len(x)):
                ans = ans*(math.sin(A[j]*(abs(x[j]-0.5)-B[j])**4)*math.cos(2*abs(x[j]-0.5)-B[j])+((abs(x[j]-0.5)-B[j])/2))
            return ans*C
        
        # defining test function
        start = time.time()
        test_func, x_1_min, x_1_max, x_2_min, x_2_max, x_3_min, x_3_max  = test_bed_2, 0, 1, 0, 1, 0, 1
    
        # generate data
        
        input_dim = 3
        x_train, z_train, x_test, z_test = make_data(input_dim, test_func, [x_1_min, x_1_max], [x_2_min, x_2_max], [x_3_min, x_3_max])
    
        '''
        Afterwards this test works with the same loaded data for each library
        '''
    
        # Reshaping the data according to library configuration
    
        pgp.load_mult_data(x_train, z_train)
    
        # Constructing the Kernel
    
        kernel_dict_input = {}
        #kernel_dict_input1 = {}
        #kernel_dict_input['Matern'] = {'lengthscale': [2,3], 'order': 1.5, 'lengthscale_bounds': '(1e-5, 1e5)'}
        kernel_dict_input['RBF'] = {'lengthscale': [1,1,1], 'lengthscale_bounds': '(1e-5, 1e5)', 'scale': 1}
        #kernel_dict_input['RatQd'] = {'lengthscale': 1 , 'power': 1.5, 'lengthscale_bounds': '(1e-5, 1e5)'}
        #kernel_dict_input1['Const'] = {'constant': 1}
        
        k = kernel.Kernel()
        k.construct(kernel_dict_input)
        
        #k1 = kernel.Kernel()
        #k1.construct(kernel_dict_input1)
        
        pgp.set_kernel (k, ard = True)
        '''
        Set ard = True for different lenghscale for different dimensions
        '''
        # Specifying the Mean Function
    
        m = mean.Mean()
        m.construct('Zero')
        pgp.set_mean (m)
    
        # Construction of the regression model
    
        pgp.init_model(noise=0.0001)
        pgp.optimize(param_opt='MLE', itr=10)
        
        # Making predictions
    
        z_postmean, z_postvar = pgp.predict_mult(x_test)
    
        # Plotting the predictions
    
        #plot(test_data, train_data, z_postmean, z_postvar)
    
        # Computing accuracy
        
        test_data = pd.DataFrame()
        test_data["z_test"] = z_test
        accuracy(test_data, z_postmean, z_postvar)
        emrmse_dict[i] = emrmse(test_data, z_postmean, z_postvar)
        pmrmse_dict[i] = pmrmse(z_postvar)
        # Contour plots
        
        #contour(x_test, z_test, z_postmean, test_func)
        end = time.time()
        time_dict[i] = end-start
        
    
    return emrmse_dict, pmrmse_dict, time_dict




