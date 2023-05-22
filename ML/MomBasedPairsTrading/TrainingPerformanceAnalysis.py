"""
This file is to run analysis on the training period results.
It groups the data frame the dimensionality reduction method and clustering method combination,
and creates a dataframe of the best average training score for each combination.
It uses the file "pickles/Cross Validated Sharpe vs Hyperparameters.pkl" as input.
It will store the result in results/train_results.csv.
"""

from MLPairsTrading import *


if __name__ == "__main__":
    print('start')

    with open('pickles/Cross Validated Sharpe vs Hyperparameters.pkl', 'rb') as handle:
        performance_df = pickle.load(handle)
    sorted_df = performance_df.sort_values(by=['AvgScore'], ascending=False)
    sorted_df = sorted_df.reset_index(drop=True)
    sorted_df = sorted_df.drop(columns=['CV2Hyperparameters', 'CV3Hyperparameters', 'CV4Hyperparameters'])
    sorted_df = sorted_df.rename(columns={'CV1Hyperparameters': 'Hyperparameters'})
    temp_df = pd.DataFrame(sorted_df['Hyperparameters'].tolist(), columns=['DimRedMethod', 'DimRedParams', 'ClusterMethod', 'ClusterParams'])
    sorted_df = pd.concat([sorted_df, temp_df], axis=1)
    best_df = sorted_df.loc[sorted_df.groupby(['DimRedMethod', 'ClusterMethod'])['AvgScore'].idxmax()]
    best_df = best_df.reset_index(drop=True)
    best_df.to_csv('results/train_results.csv')
    print('end')
