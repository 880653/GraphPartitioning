import numpy as np
import random


def chargeMatrix():
    m = np.loadtxt('Examples/Adibide1.txt', skiprows=1)
    n = int(np.loadtxt('Examples/Adibide1.txt', max_rows = 1))
    print("Having the neighbour matrix:\n",m)
    return m, n

def randoms(size):
    # create a vector of 1s, and generate random positions where to put 0s
    v = np.ones(size, dtype=int)
    positions = random.sample(range(0, size), int(size/2))
    v[positions] = 0
    
    print("and the random initial solution:\n", v)
    return v

def randomGreedy(m, size):
    v = np.full(size, -1)
    min = np.min(m[np.nonzero(m)])
    index = np.where(m == min)
    v[index[0][0]] = 0
    v[index[0][1]] = 1
    
    probabilityMatrix(m, size, v)
    
    print(v)
    return v

def probabilityMatrix(m, n, v):
    probM = np.empty([n, n])
    for i in range(n):
        for j in range(i, n):
            if((v[i] != -1) & (v[j] != -1)):
                break
            if(v[i] == -1):
                #probM[i][j] = sum(0)/sum(1)
                print("azaroak")
    return probM
            

def InitialObjectiveFunctionValue(n, v, m):
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

def ObjectiveFunctionValue(i, j, new, actualValue, m, n):
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
    m, n = chargeMatrix()
    actualSolution = randoms(n)
    actualValue = InitialObjectiveFunctionValue(n, actualSolution, m)
    newSolution=[]
    for i in range(0, n):
        for j in range(i+1, n):
            if (actualSolution[i] != actualSolution[j]):
                newSolution = swap(actualSolution, i, j)
                newValue = ObjectiveFunctionValue(i, j, newSolution, actualValue, m, n)
                if(newValue < actualValue):
                    actualSolution = newSolution
                    actualValue = newValue
    print("But if we look for local optimum we get the solution \n", newSolution)
    print("and the sum:", actualValue)
    return actualSolution, actualValue

########################          EXECUTION          ########################

GraphPartitioning()

m,n = chargeMatrix()
randomGreedy(m,n)
