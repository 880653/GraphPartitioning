import numpy as np
import random


def ObjectiveFunctionValue(n,m,v):
    sum = 0
    for i in range (n):
        for j in range (i, n):
            sum += (v[i]*(1-v[j]) + v[j]*(1-v[i]))*m[i][j]
    print("We get the initial sum:", sum, "\n")
    return sum

def randoms():
    # read the matrix provided and its size
    size = int(np.loadtxt('Examples/Adibide1.txt', max_rows = 1))
    m = np.loadtxt('Examples/Adibide1.txt', skiprows=1)
    
    # create a vector of 1s, and generate random positions where to put 0s
    v = np.ones(size, dtype=int)
    positions = random.sample(range(0, size), int(size/2))
    v[positions] = 0
    
    print("Having the neighbour matrix:\n",m)
    print("and the random solution:\n", v)
    return size,m,v

def swap(actual, i, j):
    aux = actual.copy()
    aux[i] = actual[j]
    aux[j] = actual[i]
    return aux

def ObjectiveFunctionValueWithActual(i, j, actual, new, actualValue):
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


# def NewFunctionValue(actualSolution, newSolution, actualValue):
#     newValue = actualValue.copy()
#     changes = (actualSolution - newSolution).copy()
#     print(changes, "changes")
    
#     for i in changes:
#         if ((changes[i]!=1) & (changes[i]!=-1)):
#             if(actualSolution[i] == 1):
#                 pass
#             elif(actualSolution[i] == -1):
#                 pass
#             else:
#                 pass
#     return newValue
            
    
    

def totalNeighbourhood(n, actualSolution, actualValue):
    newSolution=[]
    for i in range(0, n):
        for j in range(i+1, n):
            if (actualSolution[i] != actualSolution[j]):
                newSolution = swap(actualSolution, i, j)
                newValue = ObjectiveFunctionValueWithActual(i, j, actualSolution, newSolution, actualValue)
                if(newValue < actualValue):
                    actualSolution = newSolution
                    actualValue = newValue
    return actualSolution, actualValue

###############################################################################

n,m,v = randoms()
value = ObjectiveFunctionValue(n,m,v)


solution, sum2 = totalNeighbourhood(n, v, value)
print("But if we look for local optimum we get the solution \n", solution)
print("and the sum:", sum2)

