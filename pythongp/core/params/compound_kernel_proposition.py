class CompoundKernel(Kernel):
    """Kernel which is composed of a set of other kernels.
    .. versionadded:: 0.18
    Parameters
    ----------
    kernels : list of Kernel objects
        The other kernels
    """

    def __init__(self, kernels):
        self.kernels = kernels

    def __add__(self, b):
        if not isinstance(b, Kernel):
            return Sum(self, ConstantKernel(b))
        self.kernels = [kernels, '+', b]

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        self.kernels = [kernels, '*', b]

    def __pow__(self, b):
        rself.kernels = [kernels, '**', b]

    #def __eq__(self, b):
    #    if type(self) != type(b):
    #        return False
    #    params_a = self.get_params()
    #    params_b = b.get_params()
    #    for key in set(list(params_a.keys()) + list(params_b.keys())):
    #        if np.any(params_a.get(key, None) != params_b.get(key, None)):
    #            return False
    #    return True
