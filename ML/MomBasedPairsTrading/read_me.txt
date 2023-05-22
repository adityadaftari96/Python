For creating the testing period plots, you want to basically run the TestingPerformanceAnalysis.py and/or TestingBestPerformers.py

Below are the filenames and their descriptions, in the order that they were run for the first time to create the required files to be used as input by the next file.
The output files are included in this folder, so you can run TestingPerformanceAnalysis.py and/or TestingBestPerformers.py directly.

1. MLPairsTrading.py: This file contains all the custom functions needed for this project. It does not run anything.

2. Training.py:
    This file is the run file for training the hyperparameters of the dimensionality reduction and clustering algorithms.
    The training period is from the start date till 2018-12-31. Testing period is after that.
    We split training performance evaluation into 4 equal folds named CV1, CV2, CV3, CV4.
    We then look at average performance across the 4 folds. We run a brute force hyperparameter tuning across 40000+ combinations
    of hyperparameters across different dimensionality reduction and clustering algorithms.
    From this training we find the best hyperparameters for each of the 15 combinations shown.
    These combinations will be tested in the testing period.
    It uses the file "pickles/fundamentals.pkl" as input.
    The results are dumped in "pickles/Cross Validated Sharpe vs Hyperparameters.pkl"

3. Testing.py:
    This file is for adhoc testing to test the performance of the strategy using a given hyperparameters.
    It uses the file "pickles/fundamentals.pkl" as input.

4. TrainingPerformanceAnalysis.py:
    This file is to run analysis on the training period results.
    It groups the data frame the dimensionality reduction method and clustering method combination,
    and creates a dataframe of the best average training score for each combination.
    It uses the file "pickles/Cross Validated Sharpe vs Hyperparameters.pkl" as input.
    It will store the result in results/train_results.csv.

5. TestingPerformanceAnalysis.py:
    This is the run file for testing the performance in the testing period for the 15 combinations of dimensionality reduction
    and clustering algorithms. It dumps the results in "pickles/comb_test_perf_case2.pkl" and also plots the results for
    the 15 combination strategies.
    It uses the files "pickles/fundamentals.pkl" and "pickles/Cross Validated Sharpe vs Hyperparameters.pkl" as input.

6. TestingBestPerformers.py:
    This file uses the output of TestingPerformanceAnalysis.py, which is the file "pickles/comb_test_perf_case2.pkl", and
    then selects 3 best combinations and plot those results.
    It used the file "pickles/comb_test_perf_case2.pkl" as input.