'''


'''
from pythongp.core.params import kernel
import openturns as ot
import numpy as np

class openturns_wrapper():

    def __init__(self):

        # library
        self.library = 'openturns'

        # model definition
        self.model = None

        self.mean_function = None

        self.kernel_function = None
        
        self.nugget = None

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

        x = ot.Sample([[i] for i in list(dataframe["x"])])
        if data_type == 'train':
            z = ot.Sample([[i] for i in list(dataframe["z_train"])])

            
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
        self.x_train = ot.Sample(x_train)
        self.z_train = ot.Sample(np.reshape(z_train,(len(self.x_train),1)))
        self.input_dim = x_train.shape[1]    

    
    def get_kernel_function(self, kernel, name):
        '''
        kernel : dictionary of parameters
        name : name of the kernel
        '''
        if self.input_dim == 1:
            if name == 'Matern':
                return (ot.MaternModel([float(kernel[name]['lengthscale'])], [kernel[name]['scale']], 
                                                  float(kernel[name]['order'])))
            elif name == 'RBF':
                return (ot.SquaredExponential([float(kernel[name]['lengthscale'])], [kernel[name]['scale']]))
            elif name == 'White':
                return ('Not sure whether this library supports the specified kernel type')
            elif name == 'Const':
                return ('Not sure whether this library supports the specified kernel type')
            elif name == 'RatQd':
                return ('Not sure whether this library supports the specified kernel type')       
        else:
            if name == 'Matern':
                return (ot.MaternModel(kernel[name]['lengthscale'], [kernel[name]['scale']], float(kernel[name]['order'])))
            elif name == 'RBF':
                return (ot.SquaredExponential(kernel[name]['lengthscale'], [kernel[name]['scale']]))
            elif name == 'White':
                return ('Not sure whether this library supports the specified kernel type')
            elif name == 'Const':
                return ('Not sure whether this library supports the specified kernel type')
            elif name == 'RatQd':
                return ('Not sure whether this library supports the specified kernel type')       
    
    def set_mean(self, mean):

        '''
        This function constructs the mean function
        takes the following argument:
            -> mean_type
        '''

        if mean.mean_type == 'Linear':
            self.mean_function = ot.LinearBasisFactory(self.input_dim).build()
        elif mean.mean_type == 'Constant':
            self.mean_function = ot.ConstantBasisFactory(self.input_dim).build()
        elif mean.mean_type == 'Quadratic':
            self.mean_function = ot.QuadraticBasisFactory(self.input_dim).build()            
        elif mean.mean_type == 'Zero':
            self.mean_function = ot.Basis()
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
         #if compound_kernel.operation == '+':
          # return (self.get_kernel(compound_kernel.left) + self.get_kernel(compound_kernel.right))
         #elif compound_kernel.operation == '*':
          # return (self.get_kernel(compound_kernel.left) * self.get_kernel(compound_kernel.right))
         #elif compound_kernel.operation == '**':
          # return (self.get_kernel(compound_kernel.left) ** self.get_kernel(compound_kernel.right))
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
        self.model = ot.KrigingAlgorithm(self.x_train, self.z_train, self.kernel_function, self.mean_function, False)
        self.model.setNoise([self.nugget]*len(self.x_train))
        
    def optimize(self, param_opt, itr):
        
        if param_opt == 'MLE':
            self.model.setOptimizeParameters(optimizeParameters = True)
        elif param_opt == 'Not_optimize':
            self.model.setOptimizeParameters(optimizeParameters = False)
        else:
            return ("This library does not support the specified Parameter optimizer")

        print(self.kernel_function.getFullParameterDescription())
        print("parameter before optimization : ",self.kernel_function.getFullParameter())

        self.model.run()

        result = self.model.getResult()
        print("parameter after optimization : \n",result.getCovarianceModel())
        print("Nugget", self.model.getNoise())
        self.model = result


    def predict(self, dataframe):

        '''
        This function predicts for the test data
        '''
        self.test_dataframe = dataframe

        self.x_test, _ = self.dftoxz(self.test_dataframe, 'test')

        if type(self.model) == str:
            return

        self.z_postmean  = np.array(self.model.getConditionalMean(self.x_test))
        self.z_postvar = np.sqrt(np.add(np.diag(np.array(self.model.getConditionalCovariance(self.x_test))), self.nugget))
        #self.z_postvar = np.sqrt(np.add(self.z_postvar, self.nugget))

        return self.z_postmean, self.z_postvar
    
    def predict_mult(self, x_test):
        '''
        This function predicts for the test data
        '''
        self.x_test = ot.Sample(x_test)
        
        if type(self.model) == str:
            return
        
        self.z_postmean  = np.array(self.model.getConditionalMean(self.x_test))
        self.z_postvar = np.sqrt(np.add(np.diag(np.array(self.model.getConditionalCovariance(self.x_test))), self.nugget))

        return self.z_postmean, self.z_postvar


    def evaluate_ker(self, x, y):
        return NotImplemented