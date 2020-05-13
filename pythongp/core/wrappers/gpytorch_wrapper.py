'''
    Wrapper functions
    Author: S.B
    Description: The following code contains the necessary wrapper
    functions which implements Gaussian Process regression using
    python libraryes for a given test function

'''
import torch
import gpytorch
import numpy as np
from pythongp.core.params import kernel
from ast import literal_eval as make_tuple



class gpytorch_wrapper():

    def __init__(self):

        # library
        self.library = 'gpytorch'

        # model definition
        self.model = None
        
        self.likelihood = None

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

        x = torch.tensor(dataframe["x"])
        if data_type == 'train':
            z = torch.tensor(dataframe['z_train'])


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
        self.z_train = torch.tensor(z_train, dtype = torch.float32)
        self.x_train = torch.tensor(x_train, dtype = torch.float32)
        self.input_dim = x_train.shape[1]
        

    def get_kernel_function(self, kernel, name):
        if self.input_dim == 1:
            if name == 'Matern':
                covar =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=kernel[name]['order'],
                                                                                   eps = make_tuple(kernel[name]['lengthscale_bounds'])[0]))
                scale = kernel[name]['scale']
                covar._set_outputscale(scale)
                covar.base_kernel._set_lengthscale(kernel[name]['lengthscale'])
                return covar
            elif name == 'RBF':
                covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(eps = make_tuple(kernel[name]['lengthscale_bounds'])[0]))
                scale = kernel[name]['scale']
                covar._set_outputscale(scale)
                covar.base_kernel._set_lengthscale(kernel[name]['lengthscale'])
                return covar
            elif name == 'White':
                # return (gpytorch.likelihoods.FixedNoiseGaussianLikelihood(float(kernel[name]['noiselevel'])))
                return ('Not sure whether this library supports the specified kernel type')
            elif name == 'Const':
                return ('Not sure whether this library supports the specified kernel type')
            elif name == 'RatQd':
                return ('Not sure whether this library supports the specified kernel type')
        else:
            # SKI requires a grid size hyperparameter. This util can help with that
            grid_size = gpytorch.utils.grid.choose_grid_size(self.x_train)
        
            if name == 'Matern':
                covar =  gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.ScaleKernel(
                                                    gpytorch.kernels.MaternKernel(nu=kernel[name]['order'],
                                                    eps = make_tuple(kernel[name]['lengthscale_bounds'])[0],
                                                    ard_num_dims = self.input_dim)), grid_size=grid_size, num_dims = self.input_dim)
                scale = kernel[name]['scale']
                covar.base_kernel._set_outputscale(scale)
                covar.base_kernel.base_kernel._set_lengthscale(kernel[name]['lengthscale'])
                return covar
            elif name == 'RBF':
                covar = gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.ScaleKernel(
                                                    gpytorch.kernels.RBFKernel(eps = make_tuple(kernel[name]['lengthscale_bounds'])[0], 
                                                    ard_num_dims = self.input_dim)), grid_size=grid_size, num_dims = self.input_dim)
                scale = kernel[name]['scale']
                covar.base_kernel._set_outputscale(scale)
                covar.base_kernel.base_kernel._set_lengthscale(kernel[name]['lengthscale'])
                return covar
            elif name == 'White':
                # return (gpytorch.likelihoods.FixedNoiseGaussianLikelihood(float(kernel[name]['noiselevel'])))
                return ('Not sure whether this library supports the specified kernel type')
            elif name == 'Const':
                return ('Not sure whether this library supports the specified kernel type')
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
         #  return (self.get_kernel(compound_kernel.left),self.get_kernel(compound_kernel.right))
         #elif compound_kernel.operation == '*':
         #  return (self.get_kernel(compound_kernel.left),self.get_kernel(compound_kernel.right))
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
            self.mean_function = gpytorch.means.ConstantMean()
            self.mean_function.constant = torch.nn.Parameter(torch.ones(1)*np.mean(self.train_dataframe['z_train']))
        elif mean.mean_type == 'Zero':
            self.mean_function = gpytorch.means.ZeroMean()
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

        mean_proxy = self.mean_function
        kernel_proxy = self.kernel_function

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, x_train, z_train, likelihood):
                super(ExactGPModel, self).__init__(x_train, z_train, likelihood)
                self.mean_module = mean_proxy
                self.covar_module = kernel_proxy

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        class KISSGPRegressionModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(KISSGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
                
                self.mean_module = mean_proxy
                self.covar_module = kernel_proxy
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # initialize likelihood and model
        # The most common likelihood function used for GP regression is the Gaussian likelihood
        # but in GPytorch other options like Bernoulli likelihood, Softmax likelihood etc. are available

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior = None)
        
        if self.input_dim == 1:
            self.model = ExactGPModel(self.x_train, self.z_train, self.likelihood)
        else:
            self.model = KISSGPRegressionModel(self.x_train, self.z_train, self.likelihood)
        
        self.model.likelihood.initialize(noise = noise)
        
    
    def optimize(self, param_opt, itr):

        if param_opt == 'MLE':

            # Find optimal model hyperparameters
            self.model.train()
            self.likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

            training_iter = itr
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self.model(self.x_train)
                # Calc loss and backprop gradients
                loss = -mll(output, self.z_train)
                loss.backward()
                
                if self.input_dim == 1:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                            i + 1, training_iter, loss.item(),
                            self.model.covar_module.base_kernel.lengthscale.item(),
                            self.model.likelihood.noise.item()))
                else:    
                    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))

                optimizer.step()
                

        elif param_opt != 'Not_optimize':
            return ("Not sure whether this library supports the specified Parameter optimizer")

        # Training the model
        self.model.eval()
        self.likelihood.eval()

        reg_model = [self.model, self.likelihood]
        
        if self.input_dim == 1:
            print("The hyperparameters used for prediction are :\n")
            print("kernel lengthscale : ", self.model.covar_module.base_kernel.lengthscale.item())
            print("kernel scale : ", self.model.covar_module.outputscale.item())
            print("Nugget : ",self.model.likelihood.noise.item())
        else:
            print("The hyperparameters used for prediction are :\n")
            print("kernel lengthscale : ", self.model.covar_module.base_kernel.base_kernel.lengthscale)
            print("kernel scale : ", self.model.covar_module.base_kernel.outputscale.item())
            print("Nugget : ",self.model.likelihood.noise.item())
        
        self.model = reg_model


    def predict(self, dataframe):

        '''
        This function predicts for the test data
        '''
        self.test_dataframe = dataframe

        self.x_test, _ = self.dftoxz(self.test_dataframe, 'test')

        if type(self.model) == str:
            return

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model[1](self.model[0](self.x_test))
            self.z_postvar = observed_pred.variance.numpy()
            self.z_postmean = observed_pred.mean.numpy()

        return self.z_postmean, np.sqrt(self.z_postvar)
    
    def predict_mult(self, x_test):
        '''
        This function predicts for the test data
        '''
        self.x_test = torch.tensor(x_test, dtype = torch.float32)
        
        if type(self.model) == str:
            return
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model[1](self.model[0](self.x_test))
            self.z_postvar = np.sqrt(observed_pred.variance.numpy())
            self.z_postmean = observed_pred.mean.numpy()
        return self.z_postmean, self.z_postvar


    def evaluate_ker(self, x, y):
        return NotImplemented