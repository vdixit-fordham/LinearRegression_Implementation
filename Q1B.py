# Homework Assignment - 1

import numpy as numpy
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# *************************** USER DEFINED FUNTIONS ************************************

# this function is to get the W(weighted) matrix for a training dataset
def getWeightedMatrix(xMatrix, yMatrix, lambdaValue):
	# create the identity matrix using xMatrix
	identityMatrix = numpy.identity( numpy.shape( xMatrix)[1] )
	wMatrix = (numpy.linalg.inv((xMatrix.T * xMatrix) + ( lambdaValue * identityMatrix))) * ( (xMatrix.T * yMatrix) )

	return wMatrix


# this funtion is to get the MeanSquardError (MSE)
def getMSE(xMatrix, yMatrix, wMatrix):
	estimatedYMatrix = xMatrix * wMatrix
	differenceYMatrix = yMatrix - estimatedYMatrix
	diffSquaredMatrix = numpy.square(differenceYMatrix)
	#print(numpy.sum(diffSquaredMatrix))
	
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
figure = plt.figure( figsize=(12,10) )

filePath = 'C:/Users/user/Desktop/Fall 2017 Course Material/Data Mining/Homework/Homework-1/'

# Creating list for All 6 DataSet (To use in for loop),
# This will contain the following details for the data set (In the same order)
# trainingDSFile (0) , testDSFile (1), NoOfFeatures (2), PlotTitle (3), legend1(training) (4), legend2(test) (5), color1(training) (6), color1(test) (7), SubplotIndex (8)
#ds_100_10 = ['train-100-10.csv','test-100-10.csv', 10, 'train-100-10 Vs test-100-10', 'train-100-10', 'test-100-10', 'blue', 'orange', 231]
ds_100_100 = ['train-100-100.csv' , 'test-100-100.csv', 100, 'train-100-100 Vs test-100-100', 'train-100-100', 'test-100-100', 'blue', 'orange', 311]
#ds_1000_100 = ['train-1000-100.csv', 'test-1000-100.csv', 100, 'train-1000-100 Vs test-1000-100', 'train-1000-100', 'test-1000-100', 'blue', 'orange', 233]
ds_50_1000_100 = ['train-50(1000)-100.csv', 'test-1000-100.csv', 100, 'train-50(1000)-100 Vs test-1000-100', 'train-50(1000)-100', 'test-1000-100', 'blue', 'orange', 312]
ds_100_1000_100 = ['train-100(1000)-100.csv','test-1000-100.csv', 100, 'train-100(1000)-100 Vs test-1000-100', 'train-100(1000)-100', 'test-1000-100', 'blue', 'orange', 313]
#ds_150_1000_100 = ['train-150(1000)-100.csv','test-1000-100.csv', 100, 'train-150(1000)-100 Vs test-1000-100', 'train-150(1000)-100', 'test-1000-100', 'blue', 'orange', 236]

allDSList = [ds_100_100, ds_50_1000_100, ds_100_1000_100 ]
#allDSList = [ds_100_10 ]

#print( allDSList )
for counter in range(len(allDSList)):
	print( "\n******* L2 regularized linear regression For " + allDSList[counter][0] + ' --- ' + allDSList[counter][1] + " ******* ")

	trainData= numpy.genfromtxt( allDSList[counter][0], delimiter=',', skip_header=1 )
	testData= numpy.genfromtxt( allDSList[counter][1], delimiter=',', skip_header=1 )

	xTrainData = numpy.asmatrix( trainData[:, range(0,allDSList[counter][2])] )
	xTrainDataWithDefaultOnes = addDefaultOneToDataset( xTrainData )
	yTrainData = numpy.asmatrix( trainData[:, [allDSList[counter][2]]] )
	#print(xTrainDataWithDefaultOnes)
	#print(yTrainData)
	#print('xTrainDataWithDefaultOnes')
	#print(numpy.shape(xTrainDataWithDefaultOnes))
	#print('yTrainData')
	#print(numpy.shape(yTrainData))

	xTestData = numpy.asmatrix( testData[:, range(0,allDSList[counter][2])] )
	xTestDataWithDefaultOnes = addDefaultOneToDataset( xTestData )
	yTestData = numpy.asmatrix( testData[:, [allDSList[counter][2]]] )
	#print('xTestDataWithDefaultOnes')
	#print(numpy.shape(xTestDataWithDefaultOnes))
	#print('yTestData')
	#print(numpy.shape(yTestData))

	# List of lamda values
	lambdaList = list(range(1,151))
	mseTrainArray = []
	mseTestArray = []
	wMatrix = None
	#print(lambdaList)

	for innerCounter in range(len(lambdaList)):
		#print(innerCounter)
		# Getting the W matrix using the training data
		wMatrix = getWeightedMatrix( xTrainDataWithDefaultOnes, yTrainData, lambdaList[innerCounter] )
		#print('wMatrix')
		#print(numpy.shape(wMatrix))

		# Using the same W Matrix to get the MSE for Training & Test DataSet
		mseTrainArray.append( getMSE(xTrainDataWithDefaultOnes, yTrainData, wMatrix ) )
		mseTestArray.append( getMSE(xTestDataWithDefaultOnes, yTestData, wMatrix ) )
	
	print("For Training Data Set, Least MSE values is ", numpy.min(mseTrainArray), " for lambda value = ", numpy.argmin(mseTrainArray))
	print("For Test Data Set, Least MSE values is ", numpy.min(mseTestArray), " for lambda value = ", numpy.argmin(mseTestArray))
	
	plotMSE(mseTrainArray, mseTestArray,lambdaList, allDSList[counter][3], allDSList[counter][4], allDSList[counter][5], allDSList[counter][6], allDSList[counter][7], allDSList[counter][8])

plt.subplots_adjust(hspace = 0.5, wspace = 0.5) 
plt.show()

figure.savefig("Plot_Question1B.pdf")

print("\n\n\nPlot generated successfully !!")

	
	
