# LinearRegression_Implementation
Implementation of L2 regularized linear regression, supervised machine learning algorithm.

We are given the following 3 datasets. Each dataset has a training and a test fille. Specifically, these files are:
dataset 1: train-100-10.csv test-100-10.csv
dataset 2: train-100-100.csv test-100-100.csv
dataset 3: train-1000-100.csv test-1000-100.csv
Start the experiment by creating 3 additional training files from the train-1000-100.csv by taking the first 50, 100, and 150 instances respectively. Call them: train-50(1000)-100.csv, train-100(1000)-100.csv, train-150(1000)-100.csv. The corresponding test file for
these dataset would be test-1000-100.csv and no modification is needed.

1. Implement L2 regularized linear regression algorithm with "lambda" ranging from 0 to 150 (integers only). For each of the 6 dataset, plot both the training set MSE and the testset MSE as a function of "lambda" (x-axis) in one graph. <br />
(a) For each dataset, which "lambda" value gives the least test set MSE? <br />
(b) For each of datasets 100-100, 50(1000)-100, 100(1000)-100, provide an additional graph with "lambda" ranging from 1 to 150. <br />
(c) Explain why "lambda" = 0 (i.e., no regularization) gives abnormally large MSEs for those three datasets in (b). <br />

2. From the plots in question 1, we can tell which value of "lambda" is best for each dataset once we know the test data and its labels. This is not realistic in real world applications. In this part, we use cross validation (CV) to set the value for "lambda". Implement the 10-fold CV technique to select the best "lambda" value from the training set. <br />
(a) Using CV technique, what is the best choice of "lambda" value and the corresponding test set MSE for each of the six datasets? <br />
(b) How do the values for "lambda" and MSE obtained from CV compare to the choice of "lambda" and MSE in question 1(a)? <br />
(c) What are the drawbacks of CV? <br />
(d) What are the factors afecting the performance of CV? <br />

3. Fix "lambda" = 1, 25, 150. For each of these values, plot a learning curve for the algorithm using the dataset 1000-100.csv.
Note: a learning curve plots the performance (i.e., test set MSE) as a function of the size of the training set. To produce the curve, we need to draw random subsets (of increasing sizes) and record performance (MSE) on the corresponding test set when training on these subsets. In order to get smooth curves, we should repeat the process at least 10 times and average the results.
