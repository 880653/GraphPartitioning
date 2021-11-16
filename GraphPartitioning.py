import numpy as np
import random


def ObjectiveFunctionValue ():
    n,m,v = randoms()
    sum = 0
    for i in range (n):
        for j in range (i, n):
            sum += (v[i]*(1-v[j]) + v[j]*(1-v[i]))*m[i][j]
    print("We get the sum:", sum)
    return sum

def randoms ():
    
    # read the matrix provided and its size
    size = int(np.loadtxt('Examples\Adibide1.txt', max_rows = 1))
    m = np.loadtxt('Examples\Adibide1.txt', skiprows=1)
    
    # create a vector of 1s, and generate random positions where to put 0s
    v = np.ones(size, dtype=int)
    positions = random.sample(range(0, size), int(size/2))
    v[positions] = 0
    
    print("Having the neighbour matrix:\n",m)
    print("and the solution:\n", v)
    return size,m,v


###############################################################################

ObjectiveFunctionValue()
