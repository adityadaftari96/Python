"""
This file is the run file for training the hyperparameters of the dimensionality reduction and clustering algorithms.
The training period is from the start date till 2018-12-31. Testing period is after that.
We split training performance evaluation into 4 equal folds named CV1, CV2, CV3, CV4.
We then look at average performance across the 4 folds. We run a brute force hyperparameter tuning across 40000+ combinations
of hyperparameters across different dimensionality reduction and clustering algorithms.
From this training we find the best hyperparameters for each of the 15 combinations shown.
These combinations will be tested in the testing period.
It uses the file "pickles/fundamentals.pkl" as input.
The results are dumped in "pickles/Cross Validated Sharpe vs Hyperparameters.pkl"
"""

from MLPairsTrading import *


if __name__ == "__main__":
    leverage = 8
    train_end = '2018-12-31'
    ts_cv = 4

    data_df = load_data('pickles/fundamentals.pkl')

    feature_df = generate_features(data_df)

    train_data_df, test_data_df = train_test_split_data(feature_df, train_end)

    opt_dim_red_method = None
    opt_cluster_method = None
    opt_dim_red_params = None
    opt_cluster_params = None

    start = time()

    ## Hyperparameters start
    # Dimensionality Reduction hyperparameters
    pca_params = {'explained_var': [0.99]}
    k_pca_params = {'kernel': ['rbf', 'cosine', 'poly'], 'n_components': list(range(2, 20))}
    t_sne_params = {'n_components': list(range(2, 20))}
    mds_params = {'n_components': list(range(2, 20))}
    isomap_params = {'n_components': list(range(2, 20))}

    # Clustering hyperparameters
    agglomerative_params = {'dt': list(range(1, 20))}
    dbscan_params = {'dt': list(range(1, 20)), 'min_samples': list(range(2, 20))}
    hdbscan_params = {'min_cluster_size': list(range(2, 20))}
    ## Hyperparameters end

    dim_red_params_outer_dict = {'pca': pca_params,
                                 'kernel_pca': k_pca_params,
                                 't_sne': t_sne_params,
                                 'mds': mds_params,
                                 'isomap': isomap_params}

    cluster_params_outer_dict = {'agglomerative': agglomerative_params,
                                 'dbscan': dbscan_params,
                                 'hdbscan': hdbscan_params}

    """Find the best hyperparameters in training period"""

    # Computing total number of hyperparameter combinations to try
    # sum1 = 0
    # for k1, v1 in dim_red_params_outer_dict.items():
    #     prod = 1
    #     for k2, v2 in v1.items():
    #         prod = prod*len(v2)
    #     sum1 = sum1 + prod
    # sum2 = 0
    # for k1, v1 in cluster_params_outer_dict.items():
    #     prod = 1
    #     for k2, v2 in v1.items():
    #         prod = prod * len(v2)
    #     sum2 = sum2 + prod
    # total_combinations = sum1*sum2
    # end
    # performance_arr = np.zeros((total_combinations, 2))

    train_len = len(train_data_df.index)
    train_val_len = train_len // ts_cv
    pos = 0
    performance_df = None
    for val_count in range(ts_cv):
        print("\n------CV{0}------".format(val_count+1))
        cv_train_df = train_data_df.iloc[pos: pos + train_val_len] if val_count < ts_cv - 1 else train_data_df.iloc[pos:]
        pos += train_val_len
        performance_list = []
        for dim_red_method in dim_red_params_outer_dict.keys():
            dim_red_params_dict = dim_red_params_outer_dict[dim_red_method]
            dr_param_names = list(dim_red_params_dict.keys())
            n1 = len(dr_param_names)
            dr_combination_list = list(itertools.product(*[x for x in dim_red_params_dict.values()]))

            for dr_combination in dr_combination_list:
                dim_red_params = {dr_param_names[i]: dr_combination[i] for i in range(n1)}

                pp_feature_dict, index_map = preprocess_features(cv_train_df, dim_red_method, dim_red_params)

                for cluster_method in cluster_params_outer_dict.keys():

                    cluster_params_dict = cluster_params_outer_dict[cluster_method]
                    cluster_param_names = list(cluster_params_dict.keys())
                    n2 = len(cluster_param_names)
                    cluster_combination_list = list(itertools.product(*[x for x in cluster_params_dict.values()]))

                    for cluster_combination in cluster_combination_list:
                        cluster_params = {cluster_param_names[i]: cluster_combination[i] for i in range(n2)}

                        monthly_cluster_dict = perform_clustering(pp_feature_dict, cluster_method, cluster_params)

                        monthly_positions = create_portfolio(monthly_cluster_dict, cv_train_df, index_map, leverage)

                        train_score = evaluate_portfolio_performance(monthly_positions, data_df, dim_red_method,cluster_method, leverage, ret_only_sharpe=True)
                        print(dim_red_method, dim_red_params, cluster_method, cluster_params, round(train_score, 2))
                        performance_list.append([(dim_red_method, dim_red_params, cluster_method, cluster_params), round(train_score, 4)])

        cv_performance_df = pd.DataFrame(performance_list, columns=['CV{0}Hyperparameters'.format(val_count+1), 'CV{0}Score'.format(val_count+1)])
        # cv_performance_df = cv_performance_df.set_index(['Hyperparameters'])
        performance_df = cv_performance_df if performance_df is None else performance_df.merge(cv_performance_df, how='outer', left_index=True, right_index=True)

    performance_df['AvgScore'] = performance_df.mean(axis=1)
    sorted_df = performance_df.sort_values(by=['AvgScore'], ascending=False)
    opt_dim_red_method, opt_dim_red_params, opt_cluster_method, opt_cluster_params = sorted_df['CV1Hyperparameters'].iloc[0]

    with open('pickles/Cross Validated Sharpe vs Hyperparameters.pkl', 'wb') as handle:
        pickle.dump(performance_df, handle)

    end = time()
    print("Training time = {0} minutes".format((end-start)/60.0))

    print("\n======> Optimal Hyperparameters:")
    print("Dimensionality Reduction Method = {0}".format(opt_dim_red_method))
    print("Dimensionality Reduction Params = {0}".format(opt_dim_red_params))
    print("Clustering Method = {0}".format(opt_cluster_method))
    print("Clustering Params = {0}".format(opt_cluster_params))