'''

'''
from pythongp.core.params import kernel
import numpy as np
import gpflow


class gpflow_wrapper():

    def __init__(self):

        # library
        self.library = 'gpflow'

        # model definition
        self.model = None

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

        x = np.reshape(list(dataframe["x"]), (len(dataframe["x"]), 1))
        if data_type == 'train':
            z = np.reshape(list(dataframe['z_train']), (len(dataframe['z_train']), 1))
       
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
        self.z_train = np.reshape(z_train,(len(z_train),1))
        self.x_train = x_train
        self.input_dim = x_train.shape[1]


    def get_kernel_function(self, kernel, name, ard = False):

        if name == 'Matern':
            if kernel[name]['order'] == 1.5:
                return (gpflow.kernels.Matern32(input_dim=self.input_dim, variance=kernel[name]['scale'], lengthscales=
                                                kernel[name]['lengthscale'], ARD = ard))
            elif kernel[name]['order'] == 2.5:
                return (gpflow.kernels.Matern52(input_dim=self.input_dim, variance=kernel[name]['scale'], lengthscales=
                                                kernel[name]['lengthscale'], ARD = ard))
            elif kernel[name]['order'] == 0.5:
                return (gpflow.kernels.Matern12(input_dim=self.input_dim, variance=kernel[name]['scale'], lengthscales=
                                                kernel[name]['lengthscale'], ARD = ard))
        elif name == 'RBF':
            return (gpflow.kernels.RBF(input_dim=self.input_dim, variance=kernel[name]['scale'], lengthscales=
                                       kernel[name]['lengthscale'], ARD = ard))
        elif name == 'White':
            return (gpflow.kernels.White(input_dim=self.input_dim, variance=kernel[name]['noiselevel'], ARD = ard))
        elif name == 'Const':
            return (gpflow.kernels.Constant(kernel[name]['constant'], ARD = ard))
        elif name == 'RatQd':
            return (gpflow.kernels.RationalQuadratic(input_dim=self.input_dim, variance=kernel[name]['scale'], lengthscales=
                                                     kernel[name]['lengthscale'], alpha=
                                                     kernel[name]['power'], ARD = ard))

    
    def get_kernel(self, compound_kernel, ard = False):
         '''
         Takes input the Kernel_Tree and returns the evaluated the kernel function
         '''
        
         if compound_kernel.isempty():
           return None
         if compound_kernel.leaf():
           return (self.get_kernel_function(compound_kernel.operation[0],compound_kernel.operation[1], ard = ard))
         if compound_kernel.operation == '+':
           return (gpflow.kernels.Sum([self.get_kernel(compound_kernel.left),self.get_kernel(compound_kernel.right)]))
         elif compound_kernel.operation == '*':
           return (gpflow.kernels.Product([self.get_kernel(compound_kernel.left),self.get_kernel(compound_kernel.right)]))
         return None
         
            
    def set_kernel(self, input_kernel, ard = False):
        '''
        calls the get_kernel function and sets the kernel_function in the wrapper class
        '''
        if isinstance(input_kernel, kernel.Kernel):
            input_kernel = kernel.CompoundKernel(input_kernel.show())
        self.kernel_function = self.get_kernel(input_kernel, ard = ard)
        

    def set_mean(self, mean):

        '''
        This function constructs the mean function
        takes the following argument:
            -> mean_type
        '''       

        if mean.mean_type == 'Constant':
            self.mean_function = gpflow.mean_functions.Constant(c = np.ones(1)*np.mean(self.train_dataframe['z_train']))
        elif mean.mean_type == 'Zero':
            self.mean_function = gpflow.mean_functions.Zero()
        elif mean.mean_type == 'Linear':
            self.mean_function = gpflow.mean_functions.Linear()
        else:
            self.mean_function = "Not sure whether this library supports the specified mean function"

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


        else :

            self.model = gpflow.models.GPR(self.x_train, self.z_train, kern=self.kernel_function,
                                          mean_function=self.mean_function)
            self.model.likelihood.variance = noise

            print(self.model.as_pandas_table())
            
            
    def optimize(self, param_opt, itr):

            if param_opt == 'MLE':
                gpflow.train.ScipyOptimizer().minimize(self.model)
            elif param_opt != 'Not_optimize':
                return ("Not sure whether this library supports the specified Parameter optimizer")
            print(self.model.as_pandas_table())
            

    def predict(self, dataframe):

        '''
        This function predicts for the test data
        '''
        self.test_dataframe = dataframe

        self.x_test, _ = self.dftoxz(self.test_dataframe, 'test')

        if type(self.model) == str:
            return

        else:
            self.z_postmean, self.z_postvar = self.model.predict_y(self.x_test)
        
        return self.z_postmean.reshape(-1), np.sqrt(self.z_postvar.reshape(-1))
    def predict_mult(self, x_test):

        '''
        This function predicts for the test data
        '''

        self.z_postmean, self.z_postvar = self.model.predict_y(x_test)

        return self.z_postmean.reshape(-1), np.sqrt(self.z_postvar.reshape(-1))


    def evaluate_ker(self, x, y):
        return NotImplemented