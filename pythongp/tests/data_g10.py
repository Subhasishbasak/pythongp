import numpy as np
import pandas as pd

from pythongp.test_functions import test_functions
import lhsmdu


lower = [100, 1000, 1000, 10, 10, 10, 10, 10]
higher = [10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000]
lower = np.array(lower)
higher = np.array(higher)

d = 8

doe = np.array(np.transpose(lhsmdu.sample(d,d*3)))

doe = doe*np.tile(higher - lower, (24, 1))+np.tile(lower, (24, 1))

results = test_functions.g_10(doe)

doe = pd.DataFrame(doe, columns=['x{}'.format(i) for i in range(8)])

pd.concat((doe,results), axis = 1).to_csv('/Users/sebastien/data/data_g10', sep = ';')