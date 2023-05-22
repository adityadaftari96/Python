"""
This file is for adhoc testing to test the performance of the strategy using a given hyperparameters.
It uses the file "pickles/fundamentals.pkl" as input.
"""

from MLPairsTrading import *


if __name__ == "__main__":
    leverage = 8
    train_end = '2018-12-31'
    ts_cv = 4

    data_df = load_data('pickles/fundamentals.pkl')

    feature_df = generate_features(data_df)

    train_data_df, test_data_df = train_test_split_data(feature_df, train_end)

    opt_dim_red_method, opt_dim_red_params, opt_cluster_method, opt_cluster_params = 'mds', {'n_components': 4}, 'dbscan', {'dt': 8, 'min_samples': 12}

    pp_feature_dict, index_map = preprocess_features(test_data_df, opt_dim_red_method, opt_dim_red_params)

    monthly_cluster_dict = perform_clustering(pp_feature_dict, opt_cluster_method, opt_cluster_params)

    monthly_positions = create_portfolio(monthly_cluster_dict, test_data_df, index_map, leverage)

    test_summary_df, _, _ = evaluate_portfolio_performance(monthly_positions, data_df, opt_dim_red_method,
                                                           opt_cluster_method, leverage, plot=True, save_fig=False)

    print("Test Performance Summary")
    print(test_summary_df)