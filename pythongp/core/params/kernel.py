class CompoundKernel():
      """
      Kernel which is composed of a set of other kernels.
      For a non trivial compound kernel :
      - o is the operator,
      - l is the left kernel,
      - r is the right kernel.
      For a single kernel :
      - o stores the kernel
      - l and r are None or an empty CompoundKernel
      """
      def  __init__(self,o=None,l=None,r=None):
         if o == None:
             self.operation = None
             self.left = None
             self.right = None
             self.bounds = None
             self.optimize = None
             self.name = 'Not a unit kernel'
             self.dim = None
        
         else:
             if l == None:
                 l = CompoundKernel()
             elif isinstance(l, Kernel):
                 l = CompoundKernel(l.show())
             if r == None:
                 r = CompoundKernel()
             elif isinstance(r, Kernel):
                 r = CompoundKernel(r.show())
             self.left = l
             self.right = r
             self.operation = o
    
    
          
      def isempty(self):
        if self.operation == None:
          return True
        return False
          
          
      def leaf(self):
        if self.isempty():
          return False
        if (self.left.isempty() and self.right.isempty()):
          return True
        return False
    
      
      def __str__(self):
        if self.isempty():
           return ("E")
        return("( "+str(self.left)+" "+str(self.operation)+" "+str(self.right)+" )")

      def __add__(self, kernel):
        return CompoundKernel("+", self, kernel)

      def __mul__(self, kernel):
        return CompoundKernel("*", self, kernel)

      def __pow__(self, kernel):
        return CompoundKernel("**", self, kernel)


        
    
class Kernel():
    '''
    Class defining a kernel
    '''    
    
    
    def __init__(self):

        self.name = None
        self.optimize = None
        self.kernel_dict ={}
        self.kernel_dict['Matern'] = {'lengthscale': 1, 'order': 1.5, 'lengthscale_bounds': (1e-5, 1e5), 'scale': 1}
        self.kernel_dict['RBF'] = {'lengthscale': 1, 'lengthscale_bounds': (1e-5, 1e5), 'scale': 1}
        self.kernel_dict['White'] = {'noiselevel': 1, 'lengthscale_bounds': (1e-5, 1e5), 'scale': 1}
        self.kernel_dict['Const'] = {'constant': 1, 'scale': 1}
        self.kernel_dict['RatQd'] = {'lengthscale': 1, 'power': 1, 'lengthscale_bounds': (1e-5, 1e5), 'scale': 1}
    
       
    def set_bounds(self, lower, upper):
        self.bounds = [lower, upper]
        
              
    def set_name(self, name):
        self.name = name
        

    def construct(self,*arg):
        try:
            kernel_dict_input = arg[0]
            self.name = list(kernel_dict_input.keys())[0]
            self.kernel_dict[self.name].update(kernel_dict_input[self.name])
        except IndexError:
            print("Available kernels are as follows:\n\n Sklearn : RBF, Matern, Const, " +
                  "White, Matern, RatQd\n Shogun : RBF, Const\n GPytorch : RBF, Matern\n" +
                  " GPflow : RBF, Matern, Const, White, Matern, RatQd\n GPy : RBF, Matern," +
                  " White, RatQd\n")
    
            self.name = input('Enter kernel name: ')
            for j in self.kernel_dict[self.name].keys():
                self.kernel_dict[self.name][j] = input('Enter ' + str(j) + ' of ' + str(self.name) + ' kernel :')
        return self
            
    def show(self):
        return(self.kernel_dict, self.name)

    def __add__(self, kernel):
        return CompoundKernel("+", CompoundKernel(o = self.show()), kernel)

    def __mul__(self, kernel):
        return CompoundKernel("*", CompoundKernel(o = self.show()), kernel)

    def __pow__(self, kernel):
        return CompoundKernel("**", CompoundKernel(o = self.show()), kernel)
        
        
            
        