'''

'''
from pythongp.core.params import kernel
import numpy as np
import shogun

class shogun_wrapper():

    def __init__(self):

        # library
        self.library = 'shogun'

        # model definition
        self.model = None
        
        self.likelihood = None
        
        self.inference_method = None

        self.mean_function = None

        self.kernel_function = None

        # data
        self.input_dim = 1
        self.train_dataframe = None
        self.x_train = None
        self.z_train = None
        self.z = None

        self.test_dataframe = None
        self.x_test = None
        self.z_postmean = None
        self.z_postvar = None


    def dftoxz(self, dataframe, data_type):

        x = None
        z = None
        
        x = shogun.RealFeatures(np.array(dataframe["x"]).reshape(1, len(dataframe["x"])))
        if data_type == 'train':
            z = shogun.RegressionLabels(np.array(dataframe['z_train']))

        return x, z

    def load_data(self, dataframe):
        '''
        This function re-configures the training data according to the library requirement
        '''

        self.train_dataframe = dataframe
        # Re-configuration of the data
        self.z = shogun.RealFeatures(np.array(self.train_dataframe['z_train']).reshape(1, len(self.train_dataframe["z_train"])))
        self.x_train, self.z_train = self.dftoxz(self.train_dataframe, 'train')

    def load_mult_data(self, x_train, z_train):
        '''
        This function re-configures the training data according to the library requirement
        '''
        self.input_dim = x_train.shape[1]
        self.z_train = shogun.RegressionLabels(z_train)
        self.x_train = shogun.RealFeatures(np.array(x_train).reshape(self.input_dim, len(x_train)))
        


    def get_kernel_function(self, kernel, name):
    
        if name == 'Matern':
            return ('Not sure whether this library supports the specified kernel type')
        elif name == 'RBF':
            return shogun.GaussianKernel(kernel[name]['lengthscale'])
        elif name == 'White':
            return ('Not sure whether this library supports the specified kernel type')
        elif name == 'Const':
            return shogun.ConstKernel(kernel[name]['constant'])
        elif name == 'RatQd':
            return ('Not sure whether this library supports the specified kernel type')


    def get_kernel(self, compound_kernel):
         '''
         Takes input the Kernel_Tree and returns the evaluated the kernel function
         '''
        
         if compound_kernel.isempty():
           return None
         if compound_kernel.leaf():
           return (self.get_kernel_function(compound_kernel.operation[0],compound_kernel.operation[1]))
         #if compound_kernel.operation == '+':
         #  return (self.get_kernel(compound_kernel.left) + self.get_kernel(compound_kernel.right))
         #elif compound_kernel.operation == '*':
         #  return (self.get_kernel(compound_kernel.left) * self.get_kernel(compound_kernel.right))
         #elif compound_kernel.operation == '**':
         #  return (self.get_kernel(compound_kernel.left) ** self.get_kernel(compound_kernel.right))
         return None
         
            
    def set_kernel(self, input_kernel, ard = False):
        '''
        calls the get_kernel function and sets the kernel_function in the wrapper class
        '''
        if isinstance(input_kernel, kernel.Kernel):
            input_kernel = kernel.CompoundKernel(input_kernel.show())
        self.kernel_function = self.get_kernel(input_kernel)


    def set_mean(self, mean):

        '''
        This function constructs the mean function
        takes the following argument:
            -> mean_type
        '''        


        if mean.mean_type == 'Constant':
            self.mean_function = shogun.ConstMean()
            '''
            The constant can be set if known
            Here we set the constant as the estimated sample mean
            '''
            print("Sample mean is : ",np.mean(self.train_dataframe['z_train']))
            self.mean_function.set_const(np.mean(self.train_dataframe['z_train']))
            #self.mean_function.get_mean_vector(self.z)
        elif mean.mean_type == 'Zero':
            self.mean_function = shogun.ZeroMean()
        else:
            mean.mean_function = "Not sure whether this library supports the specified mean function"

        
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
            
        else:

            # initialize likelihood and model
            # The most common likelihood function used for GP regression is the Gaussian likelihood
    
            self.likelihood = shogun.GaussianLikelihood()
    
            self.inference_method = shogun.ExactInferenceMethod(self.kernel_function, self.x_train, self.mean_function,
                                                    self.z_train, self.likelihood)
            self.model = shogun.GaussianProcessRegression(self.inference_method)
                
            
    def optimize(self, param_opt, itr):
            
        if param_opt == 'MLE':
    
                grad_criterion = shogun.GradientCriterion()
                grad = shogun.GradientEvaluation(self.model, self.x_train, self.z_train, grad_criterion)
                grad.set_function(self.inference_method)
                grad_selection = shogun.GradientModelSelection(grad)
                best_theta = grad_selection.select_model()
                best_theta.apply_to_machine(self.model)
    
        elif param_opt != 'Not_optimize':
            return ("Not sure whether this library supports the specified Parameter optimizer")
        best_width=self.kernel_function.obtain_from_generic(self.inference_method.get_kernel()).get_width()
        best_scale=self.inference_method.get_scale()
        best_sigma=self.likelihood.obtain_from_generic(self.inference_method.get_model()).get_sigma()
        print("Optimized kernel lengthscale :", best_width)
        print("Optimized kernel scaling :", best_scale)
        print("Optimized observation noise :", best_sigma)
        print("Optimized Likelihood :", self.inference_method.get_marginal_likelihood_estimate())
        # Training the model
        self.model.train()

    

    def predict(self, dataframe):

        '''
        This function predicts for the test data
        '''
        self.test_dataframe = dataframe

        self.x_test, _ = self.dftoxz(self.test_dataframe, 'test')

        if type(self.model) == str:
            return
        
        else:
            self.z_postmean = self.model.apply_regression(self.x_test)
            self.z_postvar = self.model.get_variance_vector(self.x_test)

        return self.z_postmean, np.sqrt(self.z_postvar)

    def predict_mult(self, x_test):
        '''
        This function predicts for the test data
        '''
        self.x_test = shogun.RealFeatures(np.array(x_test).reshape(self.input_dim, len(x_test)))
        
        if type(self.model) == str:
            return
        else:
            self.z_postmean = self.model.apply_regression(self.x_test)
            self.z_postvar = np.sqrt(self.model.get_variance_vector(self.x_test))

        return self.z_postmean, self.z_postvar


    def evaluate_ker(self, x, y):
        return NotImplemented