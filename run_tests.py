# coding: utf-8

from pythongp.core.wrappers import pgp_init
from pythongp.tests.tests import test_zero_accumulation
from pythongp.tests.tests import test01
from pythongp.tests.tests import test02
from pythongp.tests.tests import test_multivar

import numpy as np
#np.random.seed(0)
library_list = pgp_init.available_libraries()
#library_list = ['GPy']

for i in library_list:

    print(i)
    pgp = pgp_init.set_library(i)

    #test_zero_accumulation(pgp, n_start = 2, n_stop = 3, n_test = 10001, eps = 0, noise = 0, n_restart  = 10)
    test01(pgp)
    #test_multivar(pgp)
    


