'''


'''
from ast import literal_eval as make_tuple
from pythongp.core.params import kernel
import sklearn.gaussian_process as sklearn_gp
import numpy as np

class sklearn_wrapper():

    def __init__(self):

        # library
        self.library = 'sklearn'

        # model definition
        self.model = None
        
        self.nugget = None

        self.mean_function = None

        self.kernel_function = None

        # data
        self.input_dim = 1
        self.train_dataframe = None
        self.x_train = None
        self.z_train = None

        self.test_dataframe = None
        self.x_test = None
        self.z_postmean = None
        self.z_postvar = None

    def dftoxz(self, dataframe, data_type):

        x = None
        z = None

        x = np.atleast_2d(dataframe["x"]).T
        if data_type == 'train':
            z = dataframe["z_train"]


        return x, z

    def load_data(self, dataframe):
        '''
        This function re-configures the training data according to the library requirement
        '''

        self.train_dataframe = dataframe
        # Re-configuration of the data

        self.x_train, self.z_train = self.dftoxz(self.train_dataframe, 'train')
    
    def load_mult_data(self, x_train, z_train):
        '''
        This function re-configures the training data according to the library requirement
        '''
        self.z_train = z_train
        self.x_train = x_train 
        self.input_dim = x_train.shape[1]

    def get_kernel_function(self, kernel, name):
        '''
        kernel : dictionary of parameters
        name : name of the kernel
        '''
  
        if name == 'Matern':
            return (sklearn_gp.kernels.Matern(length_scale = kernel[name]['lengthscale'], 
                                              nu = kernel[name]['order'], length_scale_bounds
                                              = make_tuple(kernel[name]['lengthscale_bounds'])) * 
                                              sklearn_gp.kernels.ConstantKernel(constant_value = kernel[name]['scale']))
        elif name == 'RBF': 
            return (sklearn_gp.kernels.RBF(length_scale = kernel[name]['lengthscale'], 
                                           length_scale_bounds = make_tuple(kernel[name]['lengthscale_bounds'])) * 
                                              sklearn_gp.kernels.ConstantKernel(constant_value = kernel[name]['scale']))
        elif name == 'White':
            return (sklearn_gp.kernels.WhiteKernel(noise_level = kernel[name]['noiselevel']) * 
                                              sklearn_gp.kernels.ConstantKernel(constant_value = kernel[name]['scale']))
        elif name == 'Const':
            return (sklearn_gp.kernels.ConstantKernel(constant_value = kernel[name]['constant']) * 
                                              sklearn_gp.kernels.ConstantKernel(constant_value = kernel[name]['scale']))
        elif name == 'RatQd':
            return (sklearn_gp.kernels.RationalQuadratic(length_scale=kernel[name]['lengthscale'], 
                                                         alpha = kernel[name]['power'], length_scale_bounds
                                                         = make_tuple(kernel[name]['lengthscale_bounds'])) * 
                                              sklearn_gp.kernels.ConstantKernel(constant_value = kernel[name]['scale']))

    def set_mean(self, mean):

        '''
        This function constructs the mean function
        takes the following argument:
            -> mean_type
        '''

        if mean.mean_type == 'Constant':
            self.mean_function = True
        elif mean.mean_type == 'Zero':
            self.mean_function = False
        else:
            self.mean_function = "This library does not support the specified mean function"

    

    def get_kernel(self, compound_kernel):
         '''
         Takes input the Kernel_Tree and returns the evaluated the kernel function
         '''
        
         if compound_kernel.isempty():
           return None
         if compound_kernel.leaf():
           return (self.get_kernel_function(compound_kernel.operation[0],compound_kernel.operation[1]))
         if compound_kernel.operation == '+':
           return (self.get_kernel(compound_kernel.left) + self.get_kernel(compound_kernel.right))
         elif compound_kernel.operation == '*':
           return (self.get_kernel(compound_kernel.left) * self.get_kernel(compound_kernel.right))
         elif compound_kernel.operation == '**':
           return (self.get_kernel(compound_kernel.left) ** self.get_kernel(compound_kernel.right))
         return None
         
            
    def set_kernel(self, input_kernel, ard = False):
        '''
        calls the get_kernel function and sets the kernel_function in the wrapper class
        '''
        if isinstance(input_kernel, kernel.Kernel):
            input_kernel = kernel.CompoundKernel(input_kernel.show())
        self.kernel_function = self.get_kernel(input_kernel)
        

    def init_model(self, noise):

        '''
        This function constructs the regression model
        '''

        if type(self.kernel_function) == str or type(self.mean_function) == str:
            if type(self.kernel_function) == str:
                print(self.kernel_function)
            if type(self.mean_function) == str:
                print(self.mean_function)
            self.model = 'No model'
            
        self.nugget = noise
    
    def optimize(self, param_opt, itr):
        
        if param_opt == 'MLE':
            optimizer_input = 'fmin_l_bfgs_b'
        elif param_opt == 'Not_optimize':
            optimizer_input = None
        else:
            return ("This library does not support the specified Parameter optimizer")

        self.model = sklearn_gp.GaussianProcessRegressor(kernel=self.kernel_function,
                                                                  n_restarts_optimizer=itr,
                                                                  alpha = self.nugget, optimizer=optimizer_input,
                                                                  normalize_y=self.mean_function, copy_X_train=True,
                                                                  random_state=None)
        print('Kernel hyperparameters before optimization :\n', self.model.kernel)
        self.model.fit(self.x_train, self.z_train)
        print('Kernel hyperparameters after optimization :\n', self.model.kernel_)
        print("Nuggets before optimization :\n", self.model.alpha)
        print("Likelihoood after optimization :\n", self.model.log_marginal_likelihood_value_)


    def predict(self, dataframe):

        '''
        This function predicts for the test data
        '''
        self.test_dataframe = dataframe

        self.x_test, _ = self.dftoxz(self.test_dataframe, 'test')

        if type(self.model) == str:
            return

        self.z_postmean, self.z_postvar = self.model.predict(self.x_test, return_std=True)
        
        # To predict with the noise we need to add the likelihood variance to the predicted posterior variance and take squareroot
        self.z_postvar = np.sqrt(np.add(np.square(self.z_postvar), self.nugget))
        return self.z_postmean, self.z_postvar
    
    def predict_mult(self, x_test):
        '''
        This function predicts for the test data
        '''
        self.x_test = x_test
        
        if type(self.model) == str:
            return
        
        self.z_postmean, self.z_postvar = self.model.predict(self.x_test, return_std=True)
        # To predict with the noise we need to add the likelihood variance to the predicted posterior variance and take squareroot
        self.z_postvar = np.sqrt(np.add(np.square(self.z_postvar), self.nugget))
        
        return self.z_postmean, self.z_postvar


    def evaluate_ker(self, x, y):
        return NotImplemented