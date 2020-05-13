def available_libraries():
    '''
    This function enables the user to select a library
    The available choices are displayed on the console
    '''

    libraries = ['Sklearn', 'Shogun', 'GPy', 'GPytorch', 'GPflow', 'ot']
    return libraries


def set_library(library):

    if library == 'Sklearn':
        from pythongp.core.wrappers import sklearn_wrapper
        pgp = sklearn_wrapper.sklearn_wrapper()
    elif library == 'Shogun':
        from pythongp.core.wrappers import shogun_wrapper
        pgp = shogun_wrapper.shogun_wrapper()
    elif library == 'GPy':
        from pythongp.core.wrappers import gpy_wrapper
        pgp = gpy_wrapper.gpy_wrapper()
    elif library == 'GPflow':
        from pythongp.core.wrappers import gpflow_wrapper
        pgp = gpflow_wrapper.gpflow_wrapper()
    elif library == 'GPytorch':
        from pythongp.core.wrappers import gpytorch_wrapper
        pgp = gpytorch_wrapper.gpytorch_wrapper()
    elif library == 'ot':
        from pythongp.core.wrappers import openturns_wrapper
        pgp = openturns_wrapper.openturns_wrapper()    
    else:
        raise ValueError('Unexpected library name : {}'.format(library))
    return pgp

def set_library_interactively():

    libraries = available_libraries()

    print ('Available libraries:\n')

    for i in range(0, len(libraries)):
        print(' (' + str(i) + ') - ' + libraries[i])
    i = int(input('Enter library number'))
    return set_library(libraries[i])
