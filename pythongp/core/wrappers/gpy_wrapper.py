'''
    Wrapper functions
    Author: S.B
    Description: The following code contains the necessary wrapper
    functions which implements Gaussian Process regression using
    python libraryes for a given test function

'''
import GPy
import numpy as np
from pythongp.core.params import kernel
import math
import matplotlib.pyplot as plt


class gpy_wrapper():

    def __init__(self):

        # library
        self.library = 'gpy'

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

    def load_multivariate_data(self,x, z):
        self.x_train = x
        self.z_train = z
        self.input_dim = x.shape[1]

    def load_data(self, dataframe):
        '''
        This function re-configures the training data according to the library requirement
        '''

        self.train_dataframe = dataframe
        # Re-configuration of the data

        self.x_train, self.z_train = self.dftoxz(self.train_dataframe, 'train')

    def get_kernel_function(self, kernel, name, ard):
        if name == 'Matern':
            if kernel[name]['order'] == 1.5:
                return (GPy.kern.Matern32(input_dim=self.input_dim, variance=kernel[name]['scale'], lengthscale=
                kernel[name]['lengthscale'], ARD = ard))
            elif kernel[name]['order'] == 2.5:
                return (GPy.kern.Matern52(input_dim=self.input_dim, variance=kernel[name]['scale'], lengthscale=
                kernel[name]['lengthscale'], ARD = ard))
            else:
                return ('Not sure whether this library supports the specified kernel type')
        elif name == 'RBF':
            return (GPy.kern.RBF(input_dim=self.input_dim, variance=kernel[name]['scale'], lengthscale=
            kernel[name]['lengthscale'], ARD = ard))
        elif name == 'White':
            return (GPy.kern.White(input_dim=self.input_dim, variance=kernel[name]['noiselevel']))
        elif name == 'Const':
            return (GPy.kern.src.static.Bias(input_dim=self.input_dim, variance = kernel[name]['constant'], name = 'constant'))
        elif name == 'RatQd':
            return (GPy.kern.RatQuad(input_dim=self.input_dim, variance=kernel[name]['scale'], lengthscale=
                                     kernel[name]['lengthscale'], power=
                                     kernel[name]['power'], ARD = ard))
       
    def load_mult_data(self, x_train, z_train):
        '''
        This function re-configures the training data according to the library requirement
        '''
        self.z_train = np.reshape(z_train,(len(z_train),1))
        self.x_train = x_train 
        self.input_dim = x_train.shape[1]
    
    def get_kernel(self, compound_kernel, ard):
         '''
         Takes input the Kernel_Tree and returns the evaluated the kernel function
         '''
         if compound_kernel.isempty():
           return None
         if compound_kernel.leaf():
           return (self.get_kernel_function(compound_kernel.operation[0],compound_kernel.operation[1], ard = ard))
         if compound_kernel.operation == '+':
           return (GPy.kern.Add([self.get_kernel(compound_kernel.left, ard = ard),self.get_kernel(compound_kernel.right, ard = ard)]))
         elif compound_kernel.operation == '*':
           return (GPy.kern.Prod([self.get_kernel(compound_kernel.left, ard = ard),self.get_kernel(compound_kernel.right, ard = ard)]))
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
            self.mean_function = GPy.mappings.constant.Constant(input_dim = self.x_train.shape[1], output_dim = 1, value=0.0)
            #self.mean_function = GPy.mappings.constant.Constant(input_dim = self.x_train.shape[1], output_dim = 1, value=np.mean(self.train_dataframe['z_train']))
        elif mean.mean_type != 'Zero':
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

        else:
            self.model = GPy.models.GPRegression(self.x_train, self.z_train, kernel=self.kernel_function,
                                        Y_metadata=None, normalizer=None,
                                        noise_var = noise, mean_function=self.mean_function)

            #TODO:() Make it a parameter
#            self.model.Gaussian_noise.variance.fix()
            if hasattr(self.model, 'sum'):
                self.model.sum.constant.variance.fix()

        print('\nBefore optimization : \n',self.model)
        
    
    def optimize(self, param_opt, itr):
        
        if param_opt in ['MLE', 'MLE_with_smart_init']:
            optimizer_input = True
            if param_opt == 'MLE':
                if itr <= 1:
                    self.model.optimize(messages=optimizer_input, max_iters=1000, start=None, clear_after_finish=False,
                                   ipython_notebook=True)
                else:
                    self.model.optimize_restarts(num_restarts=itr)
            else:
                grid = np.vectorize(lambda x: math.exp(x * math.log(10)))(np.arange(-5, 5, 1))
                scores = []

                zero_mean = not hasattr(self.model, 'constmap')

                for ls in grid:
                    self.set_isotropic_lengthscale(ls)

                    beta, variance = self.get_beta_and_var_from_ls(zero_mean, hasattr(self.model, 'sum'))

                    self.set_var(variance)

                    self.set_beta(beta, zero_mean)

                    scores.append(self.model._objective_grads(self.model.optimizer_array)[0])

                print("grid : {}".format(grid))
                print("scores : {}".format(scores))

                best_model_index = np.argmin(scores)

                self.set_isotropic_lengthscale(grid[best_model_index])

                #self.model.Mat52.lengthscale = [27.04, 83.38]

                beta, variance = self.get_beta_and_var_from_ls(zero_mean, hasattr(self.model, 'sum'))

                self.set_var(variance)

                self.set_beta(beta, zero_mean)

                self.model.optimize(messages=optimizer_input, max_iters=1000, start=None, clear_after_finish=False,
                               ipython_notebook=True)

            print('\nAfter optimization : \n', self.model)

            if hasattr(self.model, 'sum'):
                path = self.model.sum
            else:
                path = self.model
            if hasattr(path, 'Mat52'):
                lengthscales = path.Mat52.lengthscale
            if hasattr(path, 'Mat32'):
                lengthscales = path.Mat32.lengthscale
            if hasattr(path, 'rbf'):
                lengthscales = path.rbf.lengthscale

            print("values : {} ".format(lengthscales))
            print("\nOptimized parameters\n", self.model.param_array)

        elif param_opt != 'Not_optimize':
            return ("Not sure whether this library supports the specified Parameter optimizer")


    def plot_neg_likelihood(self):
        "Works for branin only"

        grid_1d = np.arange(0.001, 70, 1)

        x1, x2 = np.meshgrid(grid_1d, grid_1d)

        y = np.ones(x1.shape)

        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                self.model.Mat52.lengthscale = [x1[i,j], x2[i,j]]
                y[i,j] = (self.model._objective_grads(self.model.optimizer_array)[0])
        # print([(xi._default_constraint_.f, xi._default_constraint_.finv) if xi._default_constraint_ is not None else (np.identity,np.identity) for x in model.parameters for xi in x.parameters])

        print("min : {}".format(y.min()))

        plt.figure()

        fig, ax = plt.subplots()
        CS = ax.contour(x1, x2, y, 50)
        ax.clabel(CS, inline=1, fontsize=10)
        ax.set_title('Negative log likelihood')
        ax.set_xlabel('Lengthscale dimension 1')
        ax.set_ylabel('Lengthscale dimension 2')

        plt.show()

        y_1d = []
        for x in grid_1d:
           self.model.Mat52.lengthscale = [x,x]
           y_1d.append((self.model._objective_grads(self.model.optimizer_array)[0]))

        plt.figure()

        plt.semilogx()
        plt.plot(grid_1d, y_1d)
        plt.title("Negative log likelihood")
        plt.xlabel("l")
        plt.ylabel("Negative log likelihood")
        plt.show()

    def plot_neg_likelihood_var(self):
        "Works for branin only"

        var_init = self.model.Mat52.variance.values[0]

        grid_1d = np.arange(-1, 1, 0.001)

        grid_1d = [var_init * math.exp(x*math.log(10)) for x in grid_1d]

        y_1d = []
        for x in grid_1d:
           self.model.Mat52.variance = x
           y_1d.append((self.model._objective_grads(self.model.optimizer_array)[0]))

        plt.figure()

        plt.semilogx()
        plt.plot(grid_1d, y_1d)
        plt.title("Negative log likelihood : lengthscales : [{}, {}]".format(self.model.Mat52.lengthscale[0], self.model.Mat52.lengthscale[1]))
        plt.xlabel("var")
        plt.ylabel("Negative log likelihood")
        plt.vlines(var_init, ymin = min(y_1d), ymax = max(y_1d), label = 'estimated_var')
        plt.legend()
        plt.show()

    def plot_likelihood_path(self):
        param_1 = np.array([27.04301504, 83.37540132])
        param_2 = np.array([8.76182561, 21.70946319])

        grid_1d = np.arange(0,1, 0.001)

        zero_mean = not hasattr(self.model, 'constmap')

        y_1d = []

        for x in grid_1d:
            param = x * param_1 + (1 - x) * param_2
            self.model.Mat52.lengthscale = param

            beta, variance = self.get_beta_and_var_from_ls(zero_mean, hasattr(self.model, 'sum'))

            self.set_var(variance)

            self.set_beta(beta, zero_mean)

            y_1d.append(self.model._objective_grads(self.model.optimizer_array)[0])

        plt.plot(grid_1d, y_1d)
        plt.title("Negative log likelihood")
        plt.xlabel("path")
        plt.legend()
        plt.show()

    def get_beta_and_var_from_ls(self, zero_mean, is_sum):
        if not zero_mean:
            pred_matrix = np.ones(self.z_train.shape)
        else:
            pred_matrix = np.zeros(self.z_train.shape)

        if is_sum:
            return 0, ((self.z_train - self.z_train.mean())**2).mean()

        K = self.model.kern.K(self.x_train) / self.model.kern.variance.values[0]

        cpt = 0
        jit = 0
        success = False

        while cpt < 6 and not success:
            try:
                K_inv = np.linalg.inv(K)
                success = True
            except np.linalg.LinAlgError:
                if cpt == 0:
                    jit = K.diagonal().mean()
                else:
                    jit = jit * 10
                print("jit : {}".format(jit))
                K = K + np.identity(K.shape[0])
                cpt = cpt + 1

        if not zero_mean:
            beta = np.linalg.inv(pred_matrix.T @ K_inv @ pred_matrix) @ pred_matrix.T @ K_inv @ self.z_train
        else:
            beta = np.zeros([pred_matrix.shape[1], 1])

        variance = (self.z_train - pred_matrix @ beta).T @ K_inv @ (self.z_train - pred_matrix @ beta) / \
                   self.x_train.shape[0]

        if variance <= 0:
            before_var = variance
            variance = 1e-10
            print("Negative variance of {} brought back to {}.".format(before_var, variance))
        return beta, variance

    def predict(self, dataframe):

        '''
        This function predicts for the test data
        '''
        self.test_dataframe = dataframe

        self.x_test, _ = self.dftoxz(self.test_dataframe, 'test')

        if type(self.model) == str:
            return

        self.z_postmean, self.z_postvar = self.model.predict(self.x_test, include_likelihood=True)

        return self.z_postmean.reshape(-1), np.sqrt(self.z_postvar.reshape(-1))
    
    def predict_mult(self, x_test):
        '''
        This function predicts for the test data
        '''
        self.x_test = x_test
        
        if type(self.model) == str:
            return
        
        self.z_postmean, self.z_postvar = self.model.predict(self.x_test, include_likelihood=True)

        return self.z_postmean.reshape(-1), np.sqrt(self.z_postvar.reshape(-1))


    def predict_multivariate(self, x_test):

        '''
        This function predicts for the test data
        '''

        self.z_postmean, self.z_postvar = self.model.predict(x_test, include_likelihood=True)

        return self.z_postmean.reshape(-1), np.sqrt(self.z_postvar.reshape(-1))

    def set_var(self, variance):
        if hasattr(self.model, 'sum'):
            if hasattr(self.model.sum, 'Mat52'):
                self.model.sum.Mat52.variance = variance
            if hasattr(self.model.sum, 'Mat32'):
                self.model.sum.Mat32.variance = variance
            if hasattr(self.model.sum, 'rbf'):
                self.model.sum.rbf.variance = variance
        else:
            if hasattr(self.model, 'Mat52'):
                self.model.Mat52.variance = variance
            if hasattr(self.model, 'Mat32'):
                self.model.Mat32.variance = variance
            if hasattr(self.model, 'rbf'):
                self.model.rbf.variance = variance

    def set_isotropic_lengthscale(self, ls):
        if hasattr(self.model, 'sum'):
            if hasattr(self.model.sum, 'Mat52'):
                self.model.sum.Mat52.lengthscale = [ls] * self.model.sum.Mat52.lengthscale.shape[0]
            if hasattr(self.model.sum, 'Mat32'):
                self.model.sum.Mat32.lengthscale = [ls] * self.model.sum.Mat32.lengthscale.shape[0]
            if hasattr(self.model.sum, 'rbf'):
                self.model.sum.rbf.lengthscale = [ls] * self.model.sum.rbf.lengthscale.shape[0]
        else:
            if hasattr(self.model, 'Mat52'):
                self.model.Mat52.lengthscale = [ls] * self.model.Mat52.lengthscale.shape[0]
            if hasattr(self.model, 'Mat32'):
                self.model.Mat32.lengthscale = [ls] * self.model.Mat32.lengthscale.shape[0]
            if hasattr(self.model, 'rbf'):
                self.model.rbf.lengthscale = [ls] * self.model.rbf.lengthscale.shape[0]

    def set_beta(self, beta, zero_mean):
        if not zero_mean:
            self.model.constmap.C = beta

    def evaluate_ker(self, x, y):
        return self.model.kern.K(x,y)

    def Kernel_plot_GPy(self, x_min, x_max):
        x_1 = np.reshape(np.linspace(-5,5,1000),(1000,1))
        x_2 = np.reshape(np.zeros(1000),(1000,1))
        plt.plot(x_1, self.evaluate_ker(x_1, x_2)[:,0])
        plt.xlabel('$|x-y|$')
        plt.ylabel('$K(x,y)$')
