"""
This is the run file for testing the performance in the testing period for the 15 combinations of dimensionality reduction
and clustering algorithms. It dumps the results in "pickles/comb_test_perf_case2.pkl" and also plots the results for
the 15 combination strategies.
It uses the files "pickles/fundamentals.pkl" and "pickles/Cross Validated Sharpe vs Hyperparameters.pkl" as input.
"""

from MLPairsTrading import *


if __name__ == "__main__":
    print('start')
    leverage = 8
    train_end = '2018-12-31'
    ts_cv = 4

    data_df = load_data('pickles/fundamentals.pkl')

    feature_df = generate_features(data_df)

    train_data_df, test_data_df = train_test_split_data(feature_df, train_end)

    with open('pickles/Cross Validated Sharpe vs Hyperparameters.pkl', 'rb') as handle:
        performance_df = pickle.load(handle)
    performance_df = performance_df.drop(columns=['CV2Hyperparameters', 'CV3Hyperparameters', 'CV4Hyperparameters'])
    performance_df = performance_df.rename(columns={'CV1Hyperparameters': 'Hyperparameters', 'AvgCumReturn': 'AvgScore'})
    sorted_df = performance_df.sort_values(by=['AvgScore'], ascending=False)
    sorted_df = sorted_df.reset_index(drop=True)
    temp_df = pd.DataFrame(sorted_df['Hyperparameters'].tolist(), columns=['DimRedMethod', 'DimRedParams', 'ClusterMethod', 'ClusterParams'])
    sorted_df = pd.concat([sorted_df, temp_df], axis=1)
    best_train_df = sorted_df.loc[sorted_df.groupby(['DimRedMethod', 'ClusterMethod'])['AvgScore'].idxmax()]

    results_df = pd.DataFrame()
    comb_cum_ret = pd.DataFrame()
    for _, row in best_train_df.iterrows():

        opt_dim_red_method, opt_dim_red_params, opt_cluster_method, opt_cluster_params = row['Hyperparameters']

        pp_feature_dict, index_map = preprocess_features(test_data_df, opt_dim_red_method, opt_dim_red_params)

        monthly_cluster_dict = perform_clustering(pp_feature_dict, opt_cluster_method, opt_cluster_params)

        monthly_positions = create_portfolio(monthly_cluster_dict, test_data_df, index_map, leverage)

        test_summary_df, test_cum_ret, bm_cum_ret = evaluate_portfolio_performance(monthly_positions, data_df, opt_dim_red_method,
                                                            opt_cluster_method, leverage, plot=False, save_fig=False)

        row_res_df = pd.DataFrame()
        row_res_df['DimRedMethod'] = [opt_dim_red_method]
        row_res_df['ClusterMethod'] = [opt_cluster_method]
        row_res_df['DimRedParams'] = [opt_dim_red_params]
        row_res_df['ClusterParams'] = [opt_cluster_params]
        row_res_df['CumReturn(%)'] = [test_summary_df.loc['Cumulative Return', 'Strategy']]
        row_res_df['CAGR(%)'] = [test_summary_df.loc['CAGR', 'Strategy']]
        row_res_df['Ann. Vol(%)	Sharpe'] = [test_summary_df.loc['Annualized Volatility', 'Strategy']]
        row_res_df['Sharpe'] = [test_summary_df.loc['Sharpe', 'Strategy']]

        results_df = pd.concat([results_df, row_res_df])
        comb_cum_ret = pd.concat([comb_cum_ret, pd.DataFrame(test_cum_ret, columns=["{0}+{1}".format(opt_dim_red_method, opt_cluster_method)])], axis=1)

    results_df.to_csv('results/test_results.csv')
    comb_cum_ret = pd.concat([comb_cum_ret, pd.DataFrame(bm_cum_ret).rename(columns={'Close': 'S&P500'})], axis=1)
    comb_cum_ret.plot()
    plt.legend(fontsize=10, loc='upper left')
    plt.title("All combination strategies")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.show()
    with open("pickles/comb_test_perf_case2.pkl", 'wb') as handle:
        pickle.dump(comb_cum_ret, handle)
    print('end')
