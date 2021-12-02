import numpy as np
import random


def chargeMatrix():
    m = np.loadtxt('Examples/Adibide10.txt', skiprows=1)
    n = int(np.loadtxt('Examples/Adibide10.txt', max_rows = 1))
    print("Having the neighbour matrix:\n",m)
    return m, n

def chargeNewMatrix():
    file = open('Examples/G250.02', "r")
    line = file.readline()
    splitLine = line.split(" ")
    n = int(splitLine[0])
    edges = int(splitLine[1])
    m = np.zeros((n,n))
    for i in range(edges):
        line=file.readline()
        line=line.replace("\n", "")
        splitLine=line.split(" ")
        if(splitLine[0]!=""):
            for j in splitLine:
                print((splitLine))
                print(i)
                m[i, int(j)-1]=1
                m[int(j)-1, i]=1
    print("Having the neighbour matrix:\n",m)
    return m, n

def randoms(size):
    # create a vector of 1s, and generate random positions where to put 0s
    v = np.ones(size, dtype=int)
    positions = random.sample(range(0, size), int(size/2))
    v[positions] = 0
    
    #print("and the random initial solution:\n", v)
    return v

def randomGreedy(m, size):
    v = np.full(size, -1)
    mCopy = np.copy(m)
    np.fill_diagonal(mCopy, np.inf)
    minValue = np.min(mCopy)
    index = np.where(mCopy == minValue)
    v[index[0][0]] = 0
    v[index[0][1]] = 1

    i = 0
    while((np.sum(v == 1) < (size/2)) & (i < size)):
        if(v[i] == -1):
            newNodeIndex = probabilityVector(mCopy, size, v)
            v[newNodeIndex] = 1
        i += 1
        if(i == size-1):
            i = 10
    v[v == -1] = 0
    #print("We have the initial random greedy solution:", v)
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
    print("with the initial value:", sum)
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
    m, n = chargeNewMatrix()
    m, n = chargeMatrix()
    graspIterations = 5*int(n)
    graspIterations = 1
    initialSolutions = 5
    
    print("GRASP \n")
    Solution1, Value1, Times1 = grasp(graspIterations, m, n)
    print("GRASP value", Value1, "\n it was found", Times1, "times")
    #print("GRASP value", Value1, Solution1, "\n it was found", Times1, "times")
    
    print("MULTISTART \n")
    Solution2, Value2, Times2 = MultiStart(initialSolutions, graspIterations, m, n)
    print("Multistart value", Value2, "\n it was found", Times2, "times")
    #print("Multistart value", Value2, Solution2, "\n it was found", Times2, "times")
    
    print("GENETIC ALGORITHM \n")
    Solution3, Value3, Times3 = MultiStart(initialSolutions, graspIterations, m, n)
    print("Genetic value", Value3, "\n it was found", Times3, "times")
    
def Genetic():
    m, n = chargeNewMatrix()
    m, n = chargeMatrix()
    graspIterations = 1
    initialSolutions = 5
    
    print("GENETIC ALGORITHM \n")
    Solution3, Value3, Times3 = MultiStart(initialSolutions, graspIterations, m, n)
    print("Genetic value", Value3, "\n it was found", Times3, "times")

def grasp(iterations, m, n):
    
    mySolutions=[]
    myValues=[]
    for x in range(iterations):
        print("\n", x, ". iteration")
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
        #print("We found local optimum:", actualSolution)
        print("with the objective function value:", actualValue)
    bestValue=np.min(myValues)
    bestOptimum=np.where(myValues==bestValue)[0]
    return mySolutions[bestOptimum[0]], bestValue, len(bestOptimum)

def MultiStart(initialSolutions, iterations, m, n):
    
    mySolutions=[]
    myValues=[]
    for s in range(iterations):
        print("\n", s, "iteration")
        for x in range(initialSolutions):
            print("\n", x, ". solution")
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
            mySolutions.append(actualSolution)
            myValues.append(actualValue)
            #print("We found local optimum:", actualSolution)
            print("We found local optimum with the value:", actualValue)
    bestValue=np.min(myValues)
    bestOptimum=np.where(myValues==bestValue)[0]
    return mySolutions[bestOptimum[0]], bestValue, len(bestOptimum)

def GeneticAlgorithm(initialSolutions, iterations, m, n):
    mySolutions=[]
    myValues=[]
    for s in range(initialSolutions):
        newSolution = randomGreedy(m, n)
        newValue = InitialObjectiveFunctionValue(n, newSolution, m)
        mySolutions.append(newSolution)
        myValues.append(newValue)
    print(mySolutions, myValues)
    bestValue=np.min(myValues)
    bestOptimum=np.where(myValues==bestValue)[0]
    return mySolutions[bestOptimum[0]], bestValue, len(bestOptimum)


########################          EXECUTION          ########################


GraphPartitioning()

Genetic()
