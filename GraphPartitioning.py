import numpy as np
import random


def ObjectiveFunctionValue(n,m,v):
    sum = 0
    for i in range (n):
        for j in range (i, n):
            sum += (v[i]*(1-v[j]) + v[j]*(1-v[i]))*m[i][j]
    print("We get the sum:", sum)
    return sum

def randoms():
    #read the matrix provided and its size
    size = int(np.loadtxt('Examples/Adibide1.txt', max_rows = 1))
    m = np.loadtxt('Examples/Adibide1.txt', skiprows=1)
    
    #create a vector of 1s, and generate random positions where to put 0s
    v = np.ones(size, dtype=int)
    positions = random.sample(range(0, size), int(size/2))
    v[positions] = 0
    
    print("Having the neighbour matrix:\n",m)
    print("and the solution:\n", v)
    return size,m,v

def swap(actual, i, j):
    aux = actual.copy()
    aux[i] = actual[j]
    aux[j] = actual[i]
    return aux


def ObjectiveFunctionValueWithActual(actual, new, actualValue):
    value = actualValue.copy()
    #if we have a 1 or a -1, we have changed that node
    changes=actual-new
    for permutation in changes:
        if(permutation==1 or permutation==-1):
            for x in actual:
                
                if(x==actual[np.where(permutation)[0][0]]):
                    value-=m[x, np.where(permutation)[0][0]]
                else:
                    value+=m[x, np.where(permutation)[0][0]]
    print(value, "txarra")
    print(ObjectiveFunctionValue(n, m, new), "ona")
    return value


#def NewFunctionValue(actualSolution, newSolution, actualValue):
#    newValue = actualValue.copy()
#    changes = (actualSolution - newSolution).copy()
#    print(changes, "patata")
#    
#    for i in actualSolution:
#        if (changes[i]==1 )
    
    

def totalNeighbourhood(n, actualSolution, actualValue):
    newSolution=[]
    for i in range(0, n):
        for j in range(i+1, n):
            if (actualSolution[i] != actualSolution[j]):
                newSolution = swap(actualSolution, i, j)
                newValue = ObjectiveFunctionValueWithActual(actualSolution, newSolution, actualValue)
                if(newValue < actualValue):
                    actualSolution = newSolution
                    actualValue = newValue
    return actualSolution, actualValue



n,m,v = randoms()
value = ObjectiveFunctionValue(n,m,v)
print("a", value)
solution, sum2 = totalNeighbourhood(n, v, value)
print("b", sum2)
print("solution", solution)

