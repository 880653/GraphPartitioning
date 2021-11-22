import numpy as np
import random


def chargeMatrix():
    m = np.loadtxt('Examples/Adibide1.txt', skiprows=1)
    print("Having the neighbour matrix:\n",m)
    return m

def randoms(size):
    # create a vector of 1s, and generate random positions where to put 0s
    v = np.ones(size, dtype=int)
    positions = random.sample(range(0, size), int(size/2))
    v[positions] = 0
    
    print("and the random initial solution:\n", v)
    return v

def ObjectiveFunctionValue(n, v, m):
    sum = 0
    for i in range (n):
        for j in range (i, n):
            sum += (v[i]*(1-v[j]) + v[j]*(1-v[i]))*m[i][j]
    print("We get the initial sum:", sum, "\n")
    return sum

def swap(actual, i, j):
    aux = actual.copy()
    aux[i] = actual[j]
    aux[j] = actual[i]
    return aux

def ObjectiveFunctionValueWithActual(i, j, new, actualValue, m, n):
    value = actualValue.copy()

    for index in range(n):
        if((m[i, index]>0) &( index != j)):
            if((new[index] == new[i])):
                value -= m[i, index]
            else:
                value += m[i, index]
        if((m[j, index]>0) & (index != i)):
            if((new[index] == new[j])):
                value -= m[j, index]
            else:
                value += m[j, index]
        
    return value



def GraphPartitioning():
    m = chargeMatrix()
    n = len(m)
    actualSolution = randoms(n)
    actualValue = ObjectiveFunctionValue(n, actualSolution, m)
    newSolution=[]
    for i in range(0, n):
        for j in range(i+1, n):
            if (actualSolution[i] != actualSolution[j]):
                newSolution = swap(actualSolution, i, j)
                newValue = ObjectiveFunctionValueWithActual(i, j, newSolution, actualValue, m, n)
                if(newValue < actualValue):
                    actualSolution = newSolution
                    actualValue = newValue
    return actualSolution, actualValue

########################          EXECUTION          ########################

solution, value = GraphPartitioning()
print("But if we look for local optimum we get the solution \n", solution)
print("and the sum:", value)

