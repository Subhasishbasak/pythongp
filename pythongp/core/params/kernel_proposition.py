import compound_kernel_proposition as ckp

class Kernel():
    """Base class for all kernels.
    .. versionadded:: 0.18
    """

    def __init(self, name):
        self.name = name

    def get_name(self):
        return name

    def get_params(self, deep=True):
        """Get parameters of this kernel.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

    def set_params(self, **params):
        """Set the parameters of this kernel.
        The method works on simple kernels as well as on nested kernels.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.
        Returns
        -------
        self
        """

    def clone_with_theta(self, theta):
        """Returns a clone of self with given hyperparameters theta.
        Parameters
        ----------
        theta : array, shape (n_dims,)
            The hyperparameters
        """

    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""

    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""

    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.
        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.
        Returns
        -------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """

    def set_theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.
        Parameters
        ----------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """

    def bounds(self):
        """Returns the log-transformed bounds on the theta.
        Returns
        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """

    def __add__(self, b):
        if not isinstance(b, Kernel):
            return Sum(self, ConstantKernel(b))
        return ckp.CompoundKernel([self, "+",b])

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return ckp.CompoundKernel([self, "*",b])

    def __pow__(self, b):
        return ckp.CompoundKernel([self, "**",b])

    #def __eq__(self, b):
    #    if type(self) != type(b):
    #        return False
    #    params_a = self.get_params()
    #    params_b = b.get_params()
    #    for key in set(list(params_a.keys()) + list(params_b.keys())):
    #        if np.any(params_a.get(key, None) != params_b.get(key, None)):
    #            return False
    #    return True


    @abstractmethod
    def is_stationary(self):
    """Returns whether the kernel is stationary. """