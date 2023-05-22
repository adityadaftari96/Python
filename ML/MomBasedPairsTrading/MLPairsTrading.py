"""
This file contains all the custom functions needed for this project.
It does not run anything.
"""

import pandas as pd
from pandas.tseries.offsets import BDay, BMonthBegin
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pickle
import yfinance as yf
import seaborn as sns
import warnings
import itertools
from time import time

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}
warnings.filterwarnings('ignore')


def load_data(filepath='fundamentals.csv'):
    """
    Function to load, clean, and format the data.
    :param filepath: file path to load the raw or already processed data. csv data is considered raw and cleaned here.
                     pkl data is considered processed and is directly loaded and returned.
    :return: pd.DataFrame() of the cleaned and formatted data.
    """
    # Loads the data from the input filepath.
    if filepath.endswith('pkl'):
        with open(filepath, 'rb') as handle:
            final_df = pickle.load(handle)
    else:
        raw_data = pd.read_csv(filepath, low_memory=False)
        col_level_1 = raw_data.iloc[0, :][1:]
        raw_data = raw_data.iloc[1:, :]
        date_index = pd.to_datetime(raw_data.iloc[:, 0], format="%d-%m-%Y")
        raw_data = raw_data.set_index(date_index)
        raw_data = raw_data.iloc[:, 1:]

        indices = raw_data.columns
        col_level_0 = []
        for i in range(len(indices)):
            if not indices[i].startswith('Unnamed'):
                col_level_0.append(indices[i])
            else:
                col_level_0.append(col_level_0[i-1])

        col_index = pd.MultiIndex.from_arrays([col_level_0, col_level_1], names=('Index', 'Feature'))
        raw_data.columns = col_index

        raw_data = raw_data.drop("01-01-2010")
        raw_data = raw_data.drop(columns=['PX_OPEN', 'PX_HIGH', 'PX_LOW'], level=1)

        final_df = pd.DataFrame()
        for index in raw_data.columns.get_level_values(level=0).unique():
            df = raw_data.loc[:, pd.IndexSlice[index, :]].dropna(axis=1)
            if len(df.columns) == 20:
                final_df = pd.concat([final_df, df], axis=1)

        final_df = final_df.rename(columns={'PX_LAST': 'Close', 'PX_VOLUME': 'Volume'})
        final_df = final_df.astype(np.float32)
        final_df = final_df.sort_index(axis=1)
        with open(filepath, 'wb') as handle:
            pickle.dump(final_df, handle)
    return final_df


def train_test_split_data(df, train_period_end):
    """
    Split the data into train and test based on the provided train period end date.
    :param df: features data frame.
    :param train_period_end: datetime object for training period end date.
    :return: tuple of train dataframe and test dataframe.
    """
    train_df = df[df.index <= train_period_end]
    test_df = df[df.index > train_period_end]
    return train_df, test_df


def generate_features(df):
    """
    Prepare the features data frame from the cleaned and formatted data. converts input data into monthly frequency.
    Adds 48 columns for lagged month returns.
    :param df: output of load data, cleaned and formatted input data frame.
    :return: data frame consisting of monthly features.
    """
    df = df.asfreq(freq='M', method='pad')
    close_df = df.loc[:, pd.IndexSlice[:, 'Close']]
    mom_df = pd.DataFrame()
    for i in range(1, 49):
        temp_df = close_df.pct_change(periods=i).rename(columns={'Close': 'mom_{0}'.format(str(i))}, level=1)
        mom_df = pd.concat([mom_df, temp_df], axis=1)
    df = df.drop(columns=['Close'], level=1)
    df = pd.concat([df, mom_df], axis=1)
    new_index = pd.date_range(df.index.min(), df.index.max() + BDay(), freq='MS')
    df = df.reindex(new_index, method='pad')
    df = df.dropna(axis=0, how='any')
    return df


def preprocess_features(df, method, params):
    """
    Preprocess the monthly features by performing standard scalar, followed by dimensionality reduction.
    :param df: features dataframe
    :param method: dimensionality reduction method to be used.
    :param params: dictionary for parameters of the dimensionality reduction method.
    :return: tuple with first item - dictionary with keys as month begin date, and value as preprocessed features for that month.
             and second item - dictionary with keys as the equity index number and value as equity index name. Serves as a mapping when features are kept in array form.
    """
    monthly_features = dict()
    reduced_features = None
    index_map = dict()
    # sum = 0
    for date in df.index.tolist():
        xs_series = df.loc[date, :]
        x_data = xs_series.unstack(level=1)
        x_data_scaled = (x_data - x_data.mean(axis=0))/x_data.std(axis=0)

        # create mapping for index id and index name once
        i = 0
        if len(index_map.keys()) == 0:
            for index_name in x_data_scaled.index.tolist():
                index_map[i] = index_name
                i += 1

        if method.upper() == 'PCA':
            reduced_features = PCA(n_components=params['explained_var']).fit_transform(x_data_scaled)
        elif method.upper() == 'KERNEL_PCA':
            reduced_features = KernelPCA(n_components=params['n_components'], kernel=params['kernel']).fit_transform(x_data_scaled)
        elif method.upper() == 'T_SNE':
            reduced_features = TSNE(n_components=params['n_components'], random_state=42, method='exact').fit_transform(x_data_scaled)
        elif method.upper() == 'MDS':
            reduced_features = MDS(n_components=params['n_components'], random_state=42).fit_transform(x_data_scaled)
        elif method.upper() == 'ISOMAP':
            reduced_features = Isomap(n_components=params['n_components']).fit_transform(x_data_scaled)

        monthly_features[date] = reduced_features

        # Visualize in 2-D
        # features_2d = PCA(n_components=2).fit_transform(x_data_scaled)
        # plt.scatter(features_2d[:, 0], features_2d[:, 1])

        # Explained variance plot for PCA
        # pca = PCA().fit(x_data_scaled)
        # plt.plot(np.cumsum(pca.explained_variance_ratio_))
        # plt.xlabel('number of components')
        # plt.ylabel('cumulative explained variance')

        # sum to compute average number of features per month
        # sum += reduced_features.shape[1]

    # avg = sum/len(monthly_features)
    return monthly_features, index_map


def perform_agg_clustering(feature_dict, params):
    """
    Perform agglomerative clustering on monthly basis by using the monthly features dictionary.
    """
    monthly_clusters = dict()
    for date in feature_dict:
        features = feature_dict[date]
        clusterer = AgglomerativeClustering(distance_threshold=params['dt'], n_clusters=None)
        pred = clusterer.fit_predict(features)

        cluster_data = pd.DataFrame(pred).reset_index().rename(columns={0: 'cluster'})
        cluster_data = cluster_data.groupby(['cluster'])['index'].apply(list)
        monthly_clusters[date] = cluster_data

        # Plot clusters
        # plt.figure()
        # plt.scatter(features[:, 0], features[:, 1], c=pred, s=50, cmap='viridis')
        # plt.show()

        # Dendrogram
        # plt.figure()
        # Z = linkage(features)
        # dendrogram(Z)
        # plt.show()
    return monthly_clusters


def perform_dbscan_clustering(feature_dict, params):
    """
    Perform DBSCAN clustering on monthly basis by using the monthly features dictionary.
    """
    monthly_clusters = dict()
    for date in feature_dict:
        features = feature_dict[date]
        clusterer = DBSCAN(eps=params['dt'], min_samples=params['min_samples'])
        pred = clusterer.fit_predict(features)

        cluster_data = pd.DataFrame(pred).reset_index().rename(columns={0: 'cluster'})
        cluster_data = cluster_data.groupby(['cluster'])['index'].apply(list)
        monthly_clusters[date] = cluster_data

        # Plot clusters
        # plt.figure()
        # plt.scatter(features[:, 0], features[:, 1], c=pred, s=50, cmap='viridis')
        # plt.show()
    return monthly_clusters


def perform_hdbscan_clustering(feature_dict, params):
    """
    Perform HDBSCAN clustering on monthly basis by using the monthly features dictionary.
    """
    monthly_clusters = dict()
    for date in feature_dict:
        features = feature_dict[date]
        clusterer = HDBSCAN(min_cluster_size=params['min_cluster_size'], gen_min_span_tree=True)
        pred = clusterer.fit_predict(features)

        cluster_data = pd.DataFrame(pred).reset_index().rename(columns={0: 'cluster'})
        cluster_data = cluster_data.groupby(['cluster'])['index'].apply(list)
        monthly_clusters[date] = cluster_data

        # plt.figure()
        # clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
        #
        # plt.figure()
        # palette = sns.color_palette()
        # cluster_colors = [sns.desaturate(palette[col], sat)
        #                   if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
        #                   zip(clusterer.labels_, clusterer.probabilities_)]
        # plt.scatter(features.T[0], features.T[1], c=cluster_colors, **plot_kwds)
    return monthly_clusters


def perform_clustering(feature_dict, method, params):
    """
    Perform clustering on monthly basis by using the monthly features dictionary.
    :param feature_dict: dictionary of monthly features, output of preprocess features.
    :param method: method name for which clustering to be performed.
    :param params: dictionary of parameters for clustering.
    :return: dictionary with keys as month begin date, and value as clusters for each equity index.
    """
    if method.upper() == 'AGGLOMERATIVE':
        return perform_agg_clustering(feature_dict, params)

    elif method.upper() == 'DBSCAN':
        return perform_dbscan_clustering(feature_dict, params)

    elif method.upper() == 'HDBSCAN':
        return perform_hdbscan_clustering(feature_dict, params)


def create_portfolio(m_cluster_dict, f_df, index_map_dict, lev):
    """
    Creates monthly positions based on the trading strategy and monthly clustering data.
    :param m_cluster_dict: dictionary for monthly clustering data, output of perform_clustering.
    :param f_df: dataframe consisting of unprocessed features. Used to get past 1 month return data (mom_1).
    :param index_map_dict: dictionary of index number and name mapping, output from preprocess_features.
    :param lev: integer value for leverage to be used while creating portfolio.
    :return: dataframe consisting of monthly position weights for each equity index.
    """
    monthly_positions_df = pd.DataFrame()
    for date in m_cluster_dict:
        positions = pd.DataFrame()
        for cluster_data in m_cluster_dict[date].values:
            index_count = len(cluster_data)
            if index_count > 1:
                index_list = [index_map_dict[x] for x in cluster_data]
                mom_1 = f_df.loc[date, pd.IndexSlice[index_list, 'mom_1']]
                mom_1 = mom_1.droplevel(level=1)
                mom_1 = mom_1.sort_values(ascending=True)
                pair_count = index_count // 2
                long = mom_1[:pair_count].copy()
                short = mom_1[pair_count+1:].copy() if index_count % 2 != 0 else mom_1[pair_count:].copy()
                pair_map = {i: (long.index[i], short.index[i]) for i in range(pair_count)}
                diff = pd.Series(short.values - long.values)
                diff_std = diff.std()
                pairs = diff[diff > diff_std].index
                pair_count = len(pairs)
                if pair_count > 0:
                    long_index = [pair_map[x][0] for x in pairs]
                    short_index = [pair_map[x][1] for x in pairs]
                    long[:] = 1.0 / pair_count
                    short[:] = -1.0 / pair_count
                    long = long[long_index]
                    short = short[short_index]
                    trades = pd.concat([long, short], axis=0)
                    positions = pd.concat([positions, trades], axis=0)
        positions = positions.T
        positions['date'] = date
        positions = positions.set_index('date')
        monthly_positions_df = pd.concat([monthly_positions_df, positions])
    monthly_positions_df = monthly_positions_df.fillna(0)
    pos_df = monthly_positions_df.copy()
    neg_df = monthly_positions_df.copy()
    pos_df[pos_df < 0] = 0
    neg_df[neg_df > 0] = 0
    pos_df = pos_df.apply(lambda x: x / x.sum() if x.sum() != 0 else x, axis=1)
    neg_df = neg_df.apply(lambda x: -1 * x / x.sum() if x.sum() != 0 else x, axis=1)
    monthly_positions_df = pos_df + neg_df
    monthly_positions_df = monthly_positions_df * (lev / 2)
    return monthly_positions_df


def calc_hit_ratio(awr_df):
    """
    Hit Ratio for the strategy pnl.
    """
    pos_df = awr_df.copy()
    pos_df[pos_df < 0] = 0
    pos_df[pos_df > 0] = 1
    pos_trade_count = pos_df.sum().sum()

    neg_df = awr_df.copy()
    neg_df[neg_df > 0] = 0
    neg_df[neg_df < 0] = 1
    neg_trade_count = neg_df.sum().sum()

    return pos_trade_count / (pos_trade_count + neg_trade_count)


def calc_expected_profit(awr_df):
    """
    Expected Profit per $ for the strategy.
    """
    pos_df = awr_df.copy()
    pos_df[pos_df < 0] = 0
    pos_df[pos_df > 0] = 1
    trade_count = pos_df.sum().sum()

    pos_df = awr_df.copy()
    pos_df[pos_df < 0] = 0
    pf_ret = pos_df.sum(axis=1)
    pf_cum_value = (1 + pf_ret).cumprod()
    final_val = pf_cum_value[-1]

    avg_profit_per_trade = (final_val - 1) / trade_count
    exp_profit = calc_hit_ratio(awr_df) * avg_profit_per_trade
    return exp_profit


def calc_expected_loss(awr_df):
    """
    Expected Loss per $ for the strategy.
    """
    neg_df = awr_df.copy()
    neg_df[neg_df > 0] = 0
    neg_df[neg_df < 0] = 1
    trade_count = neg_df.sum().sum()

    neg_df = awr_df.copy()
    neg_df[neg_df > 0] = 0
    pf_ret = neg_df.sum(axis=1)
    pf_cum_value = (1 + pf_ret).cumprod()
    final_val = pf_cum_value[-1]

    avg_loss_per_trade = (final_val - 1) / trade_count
    exp_loss = (1 - calc_hit_ratio(awr_df)) * avg_loss_per_trade
    return exp_loss


def calc_sharpe(returns):
    """
    Sharpe ratio for given returns.
    """
    return (returns.mean() / returns.std()) * np.sqrt(12)


def calc_cagr(cum_returns):
    """
    Compounded Annual Growth Rate for given cumulative returns.
    """
    return (cum_returns[-1]) ** (12 / len(cum_returns)) - 1


def calc_ann_vol(returns):
    """
    Volatility for given returns.
    """
    return returns.std()*np.sqrt(12)


def prep_str(val, mul=1, unit='', rnd=None):
    """
    Prepare string formatting for nice printing of results.
    """
    if rnd is not None:
        return '{0}{1}'.format(round(val*mul, rnd), unit)
    else:
        return '{0}{1}'.format(val*mul, unit)


def evaluate_portfolio_performance(monthly_positions_df, data_df, dim_red_method, cluster_method, lev,
                                   benchmark_ticker='^GSPC', benchmark_name='S&P 500',
                                   ret_only_cum_ret=False, ret_only_sharpe=False,
                                   plot=False, save_fig=False):
    """
    Evaluate portfolio performance, compare to benchmark, create performance table, make plot.
    :param monthly_positions_df: dataframe containing monthly weights to be taken in each asset
    :param data_df: input data, output of load_data
    :param dim_red_method: string, name of dimensionality reduction method
    :param cluster_method: string, name of clustering method
    :param lev: integer, amount of leverage to take in the portfolio
    :param benchmark_ticker: string, name of benchmark ticker
    :param benchmark_name: string, name of the benchmark
    :param ret_only_cum_ret: boolean, whether to return only the cumulative returns of the strategy
    :param ret_only_sharpe: boolean, whether to return only sharpe ratio of the strategy, ret_only_cum_ret must be False
    :param plot: boolean, whether to plot the strategy vs benchmark cumulative returns
    :param save_fig: boolean, whether to save the plot
    :return: tuple of (performance summary dataframe, strategy cumulative returns, benchmark cumulative returns) if ret_only_cum_ret and ret_only_sharpe are False
    """
    data_df = data_df.asfreq(freq='BM', method='pad')
    ret_df = data_df.loc[:, pd.IndexSlice[:, 'Close']].droplevel(level=1, axis=1).pct_change()
    pos_df = monthly_positions_df.reindex(ret_df.index, method='pad')
    asset_wise_ret = pos_df * ret_df
    asset_wise_ret = asset_wise_ret.dropna(axis=0, how='all')

    pf_ret = asset_wise_ret.sum(axis=1)
    pf_cum_ret = (1 + pf_ret).cumprod()

    if ret_only_cum_ret:
        if len(pf_cum_ret) == 0:
            return 0.0
        else:
            return (pf_cum_ret.iloc[-1] - 1)*100

    if ret_only_sharpe:
        if len(pf_cum_ret) == 0:
            return 0.0
        else:
            return calc_sharpe(pf_ret)

    # Benchmark
    start_date = pf_cum_ret.index.min().date()
    end_date = pf_cum_ret.index.max().date()
    benchmark_df = yf.download(benchmark_ticker, start=start_date - BMonthBegin(2), end=end_date + BDay(1),
                               interval="1d")
    benchmark = benchmark_df['Close']
    benchmark = benchmark.asfreq(freq='BM', method='pad')
    benchmark_ret = benchmark.pct_change()
    benchmark_ret = benchmark_ret.reindex(pf_cum_ret.index, fill_value=0.0)
    benchmark_cum_ret = (1 + benchmark_ret).cumprod()

    # Cumulative return
    pf_val = prep_str((pf_cum_ret[-1] - 1), mul=100, unit='%', rnd=2)
    bm_val = prep_str((benchmark_cum_ret[-1] - 1), mul=100, unit='%', rnd=2)

    # CAGR
    pf_cagr = prep_str(calc_cagr(pf_cum_ret), mul=100, unit='%', rnd=2)
    bm_cagr = prep_str(calc_cagr(benchmark_cum_ret), mul=100, unit='%', rnd=2)

    # Annualized vol.
    pf_vol = prep_str(calc_ann_vol(pf_ret), mul=100, unit='%', rnd=2)
    bm_vol = prep_str(calc_ann_vol(benchmark_ret), mul=100, unit='%', rnd=2)

    # Sharpe
    pf_sharpe = prep_str(calc_sharpe(pf_ret), rnd=2)
    bm_sharpe = prep_str(calc_sharpe(benchmark_ret), rnd=2)

    # Hit ratio
    pf_hr = prep_str(calc_hit_ratio(asset_wise_ret), mul=100, unit='%', rnd=2)

    #  Expected Profit
    pf_exp_p = prep_str(calc_expected_profit(asset_wise_ret), rnd=4)

    # Expected Loss
    pf_exp_l = prep_str(calc_expected_loss(asset_wise_ret), rnd=4)

    # performance table
    res_df = pd.DataFrame({'Strategy': [pf_val, pf_cagr, pf_vol, pf_sharpe, pf_hr, pf_exp_p, pf_exp_l],
                           benchmark_name: [bm_val, bm_cagr, bm_vol, bm_sharpe, '-', '-', '-']},
                          index=['Cumulative Return', 'CAGR', 'Annualized Volatility', 'Sharpe', 'Hit Ratio',
                                 'Expected Profit per $1', 'Expected Loss per $1'])

    # plot
    if plot:
        print('\n\n')
        plt.plot(pf_cum_ret - 1)
        plt.plot(benchmark_cum_ret - 1)
        plt.legend(['Strategy', benchmark_name])
        plt.title("{0} + {1} based strategy".format(dim_red_method, cluster_method))
        plt.xlabel("Time")
        plt.ylabel("Cumulative Returns")
        if save_fig:
            plt.savefig("/Users/adityadaftari/Library/CloudStorage/OneDrive-nyu.edu/Documents/NYU MFE/Courses/Sem 2/ML/Research Project/Results/{0} + {1} based strategy.jpg".format(dim_red_method, cluster_method))
        # else:
        #     plt.show()

    return res_df, pf_cum_ret, benchmark_cum_ret
