import numpy as np
import random
from itertools import accumulate


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
    mCopy = np.copy(m)
    np.fill_diagonal(mCopy, np.inf)
    minValue = np.min(mCopy)
    index = np.where(mCopy == minValue)
    v[index[0][0]] = 0
    v[index[0][1]] = 1

    for i in range(size):
        if(np.sum(v == 1) < (size/2)):
            if(v[i] != -1):
                newNodeIndex = probabilityVector(mCopy, size, v)
                v[newNodeIndex] = 1
    v[v == -1] = 0
    print("and the random initial solution:\n", v)
    return v


def chooseNode(probabilityVector):
    randomN = random.uniform(0,1)
    acumulativeV = np.cumsum(probabilityVector)
    index = np.where(acumulativeV >= randomN)[0][0]
    return index
    
    

def probabilityVector(m, n, v):
    probV = np.empty([1, n])
    for i in range(n):
        if(v[i] == -1):
            sum1 = 0
            sum2 = 0.00001
            for j in range(n):
                if(v[j] != -1):
                    if(v[j] == 1):
                        sum1 += m[i,j]
                    else:
                        sum2 += m[i,j]
            probV[0,i] = sum1/sum2
        else:
            probV[0,i] = 0
    total = np.sum(probV)
    probV /= total
    return chooseNode(probV)
            

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
    graspIterations=10
    bestOptimum, bestValue = grasp(graspIterations)
    print("After", graspIterations, "iterations, the best local optimum was:", bestOptimum, 
        "with objective function value: ", bestValue)

def grasp(iterations):
    mySolutions=[]
    myValues=[]
    for x in range(iterations):
        actualSolution = randomGreedy(m, n)
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
        mySolutions.append(actualSolution)
        myValues.append(actualValue)
        print("Found local optimum: ", actualSolution)
        print(" with objective function value:", actualValue)
    bestValue=np.min(myValues)
    bestOptimum=np.where(myValues==bestValue)[0]
    return bestOptimum, bestValue
########################          EXECUTION          ########################

m, n = chargeMatrix()
GraphPartitioning()

# m,n = chargeMatrix()
# v = randomGreedy(m,n)

# print(InitialObjectiveFunctionValue(n, v, m), "pruebaaaaaa")