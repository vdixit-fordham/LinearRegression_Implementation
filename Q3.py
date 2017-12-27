# Homework Assignment - 1

import numpy as numpy
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from operator import add


# *************************** USER DEFINED FUNTIONS ************************************

# this function is to get the W(weighted) matrix for a training dataset
def getWeightedMatrix(xMatrix, yMatrix, lamda):
    # create the identity matrix using xMatrix
    identityMatrix = numpy.identity( numpy.shape( xMatrix)[1] )
    wMatrix = (numpy.linalg.inv((xMatrix.T * xMatrix) + ( lamda * identityMatrix))) * ( (xMatrix.T * yMatrix) )

    return wMatrix


# this funtion is to get the MeanSquardError (MSE)
def getMSE(xMatrix, yMatrix, wMatrix):
    estimatedYMatrix = xMatrix * wMatrix
    differenceYMatrix = yMatrix - estimatedYMatrix
    diffSquaredMatrix = numpy.square(differenceYMatrix)

    MSE = numpy.sum(diffSquaredMatrix) / numpy.shape( xMatrix )[0]
    return MSE

# This function is to plot the MSE for Training & Test dataset.
def plotMSE(mseTest, title, subplotIndex):
    plt.subplot(subplotIndex)
    plt.plot(mseTest)
    plt.title(title)
    plt.xlabel('Traing Set Size')
    plt.ylabel('MSE Values')
    

# This is to add default one's to the dataset.
def addDefaultOneToDataset(xdata):
    xdatawithdefaultones = numpy.insert(xdata,0,1,axis=1)
    return xdatawithdefaultones

# *************************** USER DEFINED FUNTIONS ************************************

# Ploting the figure
figure = plt.figure( figsize=(12,10) )


filePath = 'C:/Users/user/Desktop/Fall 2017 Course Material/Data Mining/Homework/Homework-1/'
# Creating list for All 6 DataSet (To use in for loop),
# This will contain the following details for the data set (In the same order)
# trainingDSFile (0) , testDSFile (1), NoOfFeatures (2), PlotTitle (3), legend1(training) (4), legend2(test) (5), color1(training) (6), color1(test) (7), SubplotIndex (8)
ds_100_10 = ['train-100-10.csv','test-100-10.csv', 10, 'train-100-10 Vs test-100-10', 'train-100-10', 'test-100-10', 'blue', 'orange', 231]
ds_100_100 = ['train-100-100.csv' , 'test-100-100.csv', 100, 'train-100-100 Vs test-100-100', 'train-100-100', 'test-100-100', 'blue', 'orange', 232]
ds_1000_100 = ['train-1000-100.csv', 'test-1000-100.csv', 100, 'train-1000-100 Vs test-1000-100', 'train-1000-100', 'test-1000-100', 'blue', 'orange', 233]
ds_50_1000_100 = ['train-50(1000)-100.csv', 'test-1000-100.csv', 100, 'train-50(1000)-100 Vs test-1000-100', 'train-50(1000)-100', 'test-1000-100', 'blue', 'orange', 234]
ds_100_1000_100 = ['train-100(1000)-100.csv','test-1000-100.csv', 100, 'train-100(1000)-100 Vs test-1000-100', 'train-100(1000)-100', 'test-1000-100', 'blue', 'orange', 235]
ds_150_1000_100 = ['train-150(1000)-100.csv','test-1000-100.csv', 100, 'train-150(1000)-100 Vs test-1000-100', 'train-150(1000)-100', 'test-1000-100', 'blue', 'orange', 236]

#allDSList = [ds_100_10, ds_100_100, ds_1000_100, ds_50_1000_100, ds_100_1000_100, ds_150_1000_100 ]
allDSList = [ds_1000_100]

# Number of times to run
numberOFRuns = 10
numberOfRandomSample = 100

subplotIndex = 131

for counter in range(len(allDSList)):
    
    trainDataComplete= numpy.genfromtxt( allDSList[counter][0], delimiter=',', skip_header=1 )
    testData= numpy.genfromtxt( allDSList[counter][1], delimiter=',', skip_header=1 )
    
    # Building the training data in the loop. (Random Number loop)

    xTestData = numpy.asmatrix( testData[:, range(0,allDSList[counter][2])] )
    xTestDataWithDefaultOnes = addDefaultOneToDataset( xTestData )
    yTestData = numpy.asmatrix( testData[:, [allDSList[counter][2]]] )
    
    lambdaList = list([1,25,150])
    
    for lambdaValue in range(len(lambdaList)):
        
        print("********************* Learning Curve For Lambda = ", lambdaList[lambdaValue], "**********************")
        print(" Running it for 10 iterations with Random data sample ")
		
        mseListSum = []
        for i in range(0,numberOFRuns):
            
            # Generating list of random numnbers
            randomNumberList = random.sample( range(1,1000), numberOfRandomSample )
            randomNumberList.sort()
            #print( randomNumberList )
            
            mseList = []
            for randomNumber in range(len(randomNumberList) ):
                
                # Taking test data based on the random number generated.
                trainData = trainDataComplete[:randomNumberList[randomNumber]]
        
                xTrainData = numpy.asmatrix( trainData[:, range(0,allDSList[counter][2])] )
                xTrainDataWithDefaultOnes = addDefaultOneToDataset( xTrainData )
                yTrainData = numpy.asmatrix( trainData[:, [allDSList[counter][2]]] )
        
                wMatrix = getWeightedMatrix( xTrainDataWithDefaultOnes, yTrainData, lambdaList[lambdaValue] )
                mse = getMSE(xTestDataWithDefaultOnes, yTestData, wMatrix )
                
                mseList.append( mse )
                
            #print("**** MSE Array for iteration = ",i, "is --- ", mseList )
            # Adding the results of all iterations in MSE Array to get the average MSE.
            mseListSum += mseList
            #print( mseListSum )
            #print("** After addition MSE Array ---- ", numpy.shape(mseListSum))
          
        mseArrayReshape = numpy.reshape( numpy.array(mseListSum) , (numberOFRuns, numberOfRandomSample) )
        avgMSEArray = numpy.divide( mseArrayReshape.sum(axis=0) , numberOFRuns ) 
        #avgMSEArray = numpy.divide( numpy.array(mseListSum), numberOFRuns)
        print("******* Avg MSE for all 10 iterations == ", avgMSEArray)
        
        plotMSE(avgMSEArray, "LC for Lambda-"+str(lambdaList[lambdaValue]), subplotIndex)
        subplotIndex += 1
            
plt.subplots_adjust(wspace = 0.6)
plt.show()

figure.savefig("Plot_Question3.pdf")
            