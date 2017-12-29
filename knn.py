import numpy as np
import math
import operator

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []

    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))

def main():
    trainingSet=[]
    testSet=[]
    predictions=[]
    classVotes = {}
    
    dataSet = np.loadtxt("haberman.data",delimiter=",")
    ndata = np.random.permutation(dataSet)
    size = len(ndata)
    nt = int(math.floor(size*0.7))
    
    trainingSet = ndata[0:nt]
    testSet = ndata[nt:size]
    
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], 3)
        
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
            sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
            
        predictions.append(sortedVotes[0][0])
    
    accuracy = getAccuracy(testSet, predictions)
    print(accuracy)

main()
