import math
import pandas as pd
import numpy as np
# Test functions

# 1D test functions

func_domain = {}
func_domain[1] = [0, 10]
func_domain[2] = [2.7, 7.5]
func_domain[3] = [-10, 10]
func_domain[4] = [1.9, 3.9]
func_domain[5] = [0, 1.2]
func_domain[6] = [0, 1.2]
func_domain[7] = [2.7, 7.5]


def f01(x):
    '''
    bound constraints taken to be [0, 10]
    '''
    return ((x) * (0.7 * x + math.sin(5 * x + 1) + 0.1 * math.sin(10 * x)))


def f02(x):
    '''
    Bound constraints: x in [2.7, 7.5]
    Global optimum: f(x)=-1.899599 for x = 5.145735
    '''
    return (math.sin(x) + math.sin((10 / 3) * x))


def f03(x):
    '''
    Bound constraints: x in [-10, 10]
    Global optimum: f(x)=-12.03124 for x = -6.7745761
    '''
    output = 0
    for i in range(1, 7):
        output = output + i * (math.sin((i + 1) * x + i)),
    return (-1) * output


def f04(x):
    '''
    Bound constraints: x in [1.9, 3.9]    
    Global optimum: f(x)=-3.85045 for x = 2.868034
    '''
    return (-1) * (16 * x ** 2 - 24 * x + 5) * math.exp(-x)


def f05(x):
    '''
    Bound constraints: x in [0, 1.2]
    Global optimum: f(x)=-1.48907 for x = 0.96609
    '''
    return (-1) * (1.4 - 3 * x) * math.sin(18 * x)


def f06(x):
    '''
    Bound constraints: x in [0, 1.2]
    Global optimum: f(x)=-0.824239 for x = 0.67956
    '''
    return (-1) * (x + math.sin(x)) * math.exp((-1) * x ** 2)


def f07(x):
    '''
    Bound constraints: x in [2.7, 7.5]
    Global optimum: f(x)=-1.6013 for x = 5.19978
    '''
    return (math.sin(x) + math.sin((10 / 3) * x) + math.log(x) - 0.84 * x + 3)


def f08(x):
    '''
    For accumulation test in zero
    '''
    return math.sin (2 * math.pi * x)+5;

# 2D test functions

func_domain_2d = {}
func_domain_2d['branin'] = [[-5,10],[0,15]]
func_domain_2d['trivial'] = [[-5,5],[-5,5]]
func_domain_2d['ackley'] = [[-5,5],[-5,5]]
func_domain_2d['eggholder'] = [[-512,512],[-512,512]]


# Branin function
def branin_1(x):
    a = 1
    b = 5.1/(4*(np.pi)**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    return pd.DataFrame({'f' : a * (x[:,1] - b*x[:,0]**2 + c*x[:,0] -r)**2 + s*(1-t)*np.vectorize(math.cos)(x[:,0]) + s})

def g_10(x):
    f = x[:,0] + x[:,1] + x[:,2]
    g1 = -1 + 0.0025*(x[:,3] + x[:,5])
    g2 = -1 + 0.0025*(-x[:,3] + x[:,4] + x[:,6])
    g3 = -1 + 0.01 * (-x[:,4] + x[:,7])
    g4 = 100*x[:,0] - x[:,0] * x[:,5] + 833.33252*x[:,3] - 83333.333
    g5 = x[:,1]*x[:,3] - x[:,1]*x[:,6] - 1250 * x[:,3] + 1250*x[:,4]
    g6 = x[:,2]*x[:,4] - x[:,2]*x[:,7] - 2500 * x[:,4] + 1250000
    return pd.DataFrame({'f': f, 'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4, 'g5': g5, 'g6': g6})

def branin(x, y):
    import math
    a=1
    b=5.1/(4*math.pi**2)
    c=5/math.pi
    r=6
    s=10
    t=1/(8*math.pi)
    ans = a*(y-b*x**2+c*x-r)**2+s*(1-t)*math.cos(x)+s
    return ans

def trivial(x,y):
    return (x**2 + y**2)


def trivial_2(x,y):
    return (x**3 + y**3)

def ackley(x, y):
    return (-20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.exp(1)+20)

def eggholder(x, y):
    return (-(y + 47)*np.sin(np.sqrt(abs((x/2) + y + 47)))-x*np.sin(np.sqrt(abs(x - y - 47))))
