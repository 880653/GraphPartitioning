import numpy as np
import random
from operator import itemgetter
from heapq import nsmallest
import timeit

def chargeMatrix(path):
    typeOfFile=path.split(".")[len(path.split("."))-1]
    if(typeOfFile=="txt"):
        print("charged matrix: ", path)
        m = np.loadtxt('Examples/Adibide1.txt', skiprows=1)
        n = int(np.loadtxt('Examples/Adibide1.txt', max_rows = 1))
        return m, n
    else:
        return chargeNewMatrix(path)

def chargeNewMatrix(path):
    print("charged matrix: ", path)
    file = open(path, "r")
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
                m[i, int(j)-1]=1
                m[int(j)-1, i]=1
    return m, n

def randoms(size):
    # create a vector of 1s, and generate random positions where to put 0s
    v = np.ones(size, dtype=int)
    positions = random.sample(range(0, size), int(size/2))
    v[positions] = 0
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
    average=np.average(myValues)
    variance=np.var(myValues)

    return mySolutions[bestOptimum[0]], bestValue, average, variance 

def MultiStart(iterations, m, n):
    
    mySolutions=[]
    myValues=[]
    for x in range(iterations):
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
    average=np.average(myValues)
    variance=np.var(myValues)
    return mySolutions[bestOptimum[0]], bestValue, average, variance

def crossIndividuals(ind1, ind2, n):
    crossPoint = random.randint(0,n)
    newInd1 = np.append(ind1[:crossPoint], ind2[crossPoint:])
    newInd2 = np.append(ind2[:crossPoint], ind1[crossPoint:])
    
    newInd1, newInd2 = crossCorrection(ind1, ind2, n)
    
    return newInd1, newInd2

def crossCorrection(ind1, ind2, n):
    while (np.sum(ind1 == 1) < (np.sum(ind1 == 0))):
        # more 0s
        zeroIndex1 = np.where(ind1 == 0)[0]
        randomIndex = random.choice(zeroIndex1)
        ind1[randomIndex] = 1
        
        oneIndex2 = np.where(ind2 == 1)[0]
        randomIndex = random.choice(oneIndex2)
        ind2[randomIndex] = 0
        
    while (np.sum(ind1 == 0) < (np.sum(ind1 == 1))):
        # more 1s
        oneIndex1 = np.where(ind1 == 1)[0]
        randomIndex = random.choice(oneIndex1)
        ind1[randomIndex] = 0
        
        zeroIndex2 = np.where(ind2 == 0)[0]
        randomIndex = random.choice(zeroIndex2)
        ind2[randomIndex] = 0
    return ind1, ind2

def mutation(ind, n):
    mutateRandom = random.random()
    if(0.01 > mutateRandom):
        rand = random.randint(0,n-1)
        ind[rand] = random.randint(0,1)
        print("after mutation", ind)
        
        ind = mutationCorrection(ind, n)
        
        print("after correction", ind)
    return ind

def mutationCorrection(ind, n):
    # more 0s
    while (np.sum(ind == 1) < (np.sum(ind == 0))):
        zeroIndex = np.where(ind == 0)[0]
        randomIndex = random.choice(zeroIndex)
        ind[randomIndex] = 1
    while (np.sum(ind == 0) < (np.sum(ind == 1))):
        oneIndex = np.where(ind == 1)[0]
        randomIndex = random.choice(oneIndex)
        ind[randomIndex] = 0
    return ind

def GeneticAlgorithm(initialSolutions, iterations, m, n):
    mySolutions=[]
    myValues=[]
    
    # generate initial values
    for s in range(initialSolutions):
        newSolution = randoms(n)
        newValue = InitialObjectiveFunctionValue(n, newSolution, m)
        mySolutions.append(newSolution)
        myValues.append(newValue)
        
    # stopping criteria: if the minimum has been the same for the last 5 iterations, stop
    stop = 0
    i = 0
    
    while(stop < 5):
        
        oldOptimum = np.min(myValues)
        
        # sort the values
        pairs = zip(mySolutions, myValues)
        result = nsmallest(5, pairs, key=itemgetter(1))
        myValues = [i[1] for i in result]
        mySolutions = [i[0] for i in result]
        
    
        for i in range(int(n/2)):
            rands = random.sample(mySolutions, 2)
            r1 = rands[0]
            r2 = rands[1]
            # cross + correct individuals
            new1, new2 = crossIndividuals(r1, r2, n)
            
            # mutate + correct individuals
            new1 = mutation(new1, n)
            new2 = mutation(new2, n)
            
            mySolutions.append(new1)
            myValues.append(InitialObjectiveFunctionValue(n, new1, m))
            mySolutions.append(new2)
            myValues.append(InitialObjectiveFunctionValue(n, new2, m))
            
        if(np.min(myValues) == oldOptimum):
            stop += 1
        else:
            stop = 0
        i += 1
        
        print("best value", np.min(myValues))

    bestValue=np.min(myValues)
    bestOptimum=np.where(myValues==bestValue)[0]
    average=np.average(myValues)
    variance=np.var(myValues)
    return mySolutions[bestOptimum[0]], bestValue, average, variance

def Experiments(iterations, path):
    m, n = chargeMatrix(path)
    initialSolutions=3
    # MULTISTART
    start = timeit.default_timer()
    print("\n MULTISTART")
    msSolution, msBest, msAverage, msVariance = MultiStart(iterations, m, n)
    stop = timeit.default_timer()
    print("Best: ", msBest, " Average: ", msAverage, " Variance: ", msVariance, " Time: ", stop-start)

    # GRASP
    start = timeit.default_timer()
    print("\n GRASP")
    grSolution, grBest, grAverage, grVariance = grasp(iterations, m, n)
    stop = timeit.default_timer()
    print("Best: ", grBest, " Average: ", grAverage, " Variance: ", grVariance, " Time: ", stop-start)

    # GENETIC ALGORITHM    
    start = timeit.default_timer()
    print("\n GENETIC ALGORITHM")
    gaSolution, gaBest, gaAverage, gaVariance = GeneticAlgorithm(initialSolutions, iterations, m, n)
    stop = timeit.default_timer()
    print("Best: ", gaBest, " Average: ", gaAverage, " Variance: ", gaVariance, " Time: ", stop-start)

########################          EXECUTION          ########################

paths=['Examples/Adibide1.txt','Examples/Adibide2.txt','Examples/Adibide3.txt','Examples/Adibide4.txt',
'Examples/Adibide5.txt','Examples/Adibide6.txt','Examples/Adibide7.txt','Examples/Adibide8.txt',
'Examples/Adibide9.txt','Examples/Adibide10.txt','Examples/G124.02', 'Examples/G124.16',
'Examples/G250.02','Examples/G250.04','Examples/G250.08','Examples/G500.005','Examples/G500.02',
'Examples/G500.04','Examples/G.sub.500','Examples/G1000.02','Examples/G1000.005','Examples/G1000.0025']
iterations = 3
for path in paths:
    Experiments(iterations, path)