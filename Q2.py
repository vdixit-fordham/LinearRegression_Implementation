# Homework Assignment - 1

import numpy as numpy
import matplotlib.pyplot as plt
import operator


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
def plotMSE(mseTraining , mseTest,lamdaList, title, trainLegend, testLegend, trainColor, testColor, subplotIndex):
    plt.subplot(subplotIndex)
    plt.plot(lamdaList,mseTraining)
    plt.plot(lamdaList,mseTest)
    plt.title(title)
    plt.gca().set_color_cycle([trainColor, testColor])
    plt.legend([trainLegend, testLegend])
    plt.xlabel('Lambda Values')
    plt.ylabel('MSE Values')
    

# This is to add default one's to the dataset.
def addDefaultOneToDataset(xdata):
    xdatawithdefaultones = numpy.insert(xdata,0,1,axis=1)
    return xdatawithdefaultones

# *************************** USER DEFINED FUNTIONS ************************************

# Ploting the figure
#plt.figure(1)

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

allDSList = [ds_100_10, ds_100_100, ds_1000_100, ds_50_1000_100, ds_100_1000_100, ds_150_1000_100 ]
#allDSList = [ds_100_10]

# Number of folds used for CV.
noOfFolds = 10

for counter in range(len(allDSList)):
    
    print('\n********** Running Cross Validation for Traing DataSet - ', allDSList[counter][0], '*************')
    
    trainData= numpy.genfromtxt( allDSList[counter][0], delimiter=',', skip_header=1 )
    testData= numpy.genfromtxt( allDSList[counter][1], delimiter=',', skip_header=1 )

    xTrainData = numpy.asmatrix( trainData[:, range(0,allDSList[counter][2])] )
    xTrainDataWithDefaultOnes = addDefaultOneToDataset( xTrainData )
    #xTrainDataWithDefaultOnes = xTrainData
    yTrainData = numpy.asmatrix( trainData[:, [allDSList[counter][2]]] )
    
    xTestData = numpy.asmatrix( testData[:, range(0,allDSList[counter][2])] )
    xTestDataWithDefaultOnes = addDefaultOneToDataset( xTestData )
    #xTestDataWithDefaultOnes = xTestData
    yTestData = numpy.asmatrix( testData[:, [allDSList[counter][2]]] )
    
    # Split the xData into 10 matrix - for 10 fold CV. This will be used as a Traing/Test set for CV.
    trainDatasplit = numpy.split(trainData, noOfFolds)
    #print(numpy.shape(trainDatasplit))

    #lambdaList = list([0,5,10,15,20,25,30,35,40,45,50])
    lambdaList = list(range(0,151))
    #print(lambdaList)
    
    #mseTrainingArray - This array will have [lambdaValue, ith Fold , mse]
    mseTestArray = []
      
    for lambdaValue in range(len(lambdaList)):
        #print("##### Running for lambda = ", lambdaList[lambdaValue], " #####")
        trainDataTmp = None
        # This will be the training set for CV ( All splits except the the one for which we are running the loop)
        trainDataForCV = None 
        # This will be the test set for CV ( The one for which we are running the loop )
        testDataForCV = None
        
        mseSumForAllFolds = 0
        # Inner loop to run for 10 fold Cross Validation
        for i in range(len(trainDatasplit)):
            #print("$$$$$ Running for Fold = ", i, " $$$$$")
            # Builing the taining data for Cross Validation. By Concatenating the splited matrix
            trainDataTmp = numpy.delete(trainDatasplit, [i], 0)
            testDataForCV = trainDatasplit[i]
            shapeOfTrainDataTmp = numpy.shape( trainDataTmp )
            trainDataForCV = numpy.reshape(trainDataTmp, ( shapeOfTrainDataTmp[0] * shapeOfTrainDataTmp[1], shapeOfTrainDataTmp[2]))
            #print("Shape of testDataForCV = ", numpy.shape(testDataForCV))
            #print("Shape of trainDataForCV = ", numpy.shape(trainDataForCV))
            
            # X Matrix (feature List from training & test data)
            xTrainDataForCV = numpy.asmatrix( trainDataForCV[ : , range(0,allDSList[counter][2])] )
            xTrainDataForCVWithDefaultOnes = addDefaultOneToDataset( xTrainDataForCV )
            #print("Shape of xTrainDataForCV = ", numpy.shape(xTrainDataForCV))
            #print("Shape of xTrainDataForCVWithDefaultOnes = ", numpy.shape(xTrainDataForCVWithDefaultOnes))
            xTestDataForCV = numpy.asmatrix( testDataForCV[ : , range(0,allDSList[counter][2])] )
            xTestDataForCVWithDefaultOnes = addDefaultOneToDataset( xTestDataForCV )
            #print("Shape of xTestDataForCV = ", numpy.shape(xTestDataForCV))
            #print("Shape of xTestDataForCVWithDefaultOnes = ", numpy.shape(xTestDataForCVWithDefaultOnes))
            
            # Y Matrix (From training & test data)
            yTrainDataForCV = numpy.asmatrix( trainDataForCV[:, [allDSList[counter][2]]] )
            yTestDataForCV = numpy.asmatrix( testDataForCV[:, [allDSList[counter][2]]] )
            #print("Shape of yTrainDataForCV = ", numpy.shape(yTrainDataForCV))
            #print("Shape of yTestDataForCV = ", numpy.shape(yTestDataForCV))
            
            # Ready to calculate W Matrix on the training data (remaing 9 folds)
            wMatrixCV = getWeightedMatrix( xTrainDataForCVWithDefaultOnes, yTrainDataForCV, lambdaList[lambdaValue] )
            
            mse = getMSE(xTestDataForCVWithDefaultOnes, yTestDataForCV, wMatrixCV )
            #print(mse)
            #print('For lambda=', lambdaList[lambdaValue] , ' and ith fold =',i, ', The MSE is = ', mse )
            mseSumForAllFolds += mse
        
        #print('Average MSE for all folds for lambda=', lambdaList[lambdaValue] , ' is - ' ,mseSumForAllFolds/noOfFolds)
        mseTestArray.append([lambdaList[lambdaValue] , mseSumForAllFolds/noOfFolds])  

    #print(type(mseTestArray))
    #print(mseTestArray)

    leastMSEIndex = min(mseTestArray, key=operator.itemgetter(1))
    #print( leastMSEIndex )
    print( "From CV, Least MSE is ", leastMSEIndex[1], " for Lambda Value = ", leastMSEIndex[0] )

    print("      Now Retrain the entire Training set to get the test set MSE using Best choice lambda (obtained From CV) = ", leastMSEIndex[0])
    
    wMatrix = getWeightedMatrix( xTrainDataWithDefaultOnes, yTrainData, leastMSEIndex[0] )
    mse = getMSE(xTestDataWithDefaultOnes, yTestData, wMatrix )
    
    print("      For best choice lambda", leastMSEIndex[0], "corresponding test set MSE is ", mse)
    print("\n")
#print(numpy.min(mseTestArray , axis=1))
#print( mseTestArray[numpy.argmin( mseTestArray[:])])