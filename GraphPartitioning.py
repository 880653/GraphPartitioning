import numpy as np
import random


def ObjectiveFunctionValue (n,l):
    m,v = randoms(n,l)
    sum = 0
    for i in range (n):
        for j in range (i, n):
            sum += (v[i]*(1-v[j]) + v[j]*(1-v[i]))*m[i][j]
    print("We get the sum:", sum)
    return sum

def randoms (size, limit):
    
    # make a random matrix with values between 0-10 and fill the diagonal with 0s
    m = np.random.randint(0,limit/2 + 1, (size, size))
    np.fill_diagonal(m, 0)
    m = m + m.T
    
    # create a vector of 1s, and generate random positions where to put 0s
    v = np.ones(size, dtype=int)
    positions = random.sample(range(0, size), int(size/2))
    v[positions] = 0
    
    print("Having the neighbour matrix:\n",m)
    print("and the solution:\n", v)
    return m,v


###############################################################################

n = int(input("Enter an even size: "))
l = int(input("Enter a limit value: "))
ObjectiveFunctionValue(n,l)