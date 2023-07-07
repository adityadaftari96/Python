"""
Run file for backtesting
"""
from Trend_Following_Trading import *


if __name__ == "__main__":

    print('start')
    start_time = time()

    base_folder = "/Users/adityadaftari/Library/CloudStorage/OneDrive-nyu.edu/Documents/NYU MFE/DeepAlpha/results/"
    version = 6.6
    output_suffix = "config_{:.1f}".format(version)

    # Period for parameter estimation with start and end date included
    start_date = datetime(1962, 1, 3)
    # start_date = datetime(2000, 1, 1)
    # start_date = datetime(2015, 1, 1)
    # end_date = datetime(2011, 12, 31)
    end_date = datetime(2023, 5, 30)

    ### hyperparameters

    # use thresholds for regime identification with cum return thresholds
    thresholds = [
        # start date, bear threshold, bull threshold
        [start_date, 0.20, 0.20],
        # [start_date, 0.60, 0.60],
        # [datetime(2017, 11, 9), 0.6, 0.6],
        [datetime(2010, 1, 1), 0.18, 0.20],
        # [datetime(2020, 1, 1), 0.17, 0.23],
    ]
    # end of cum return thresholds

    # for change point detection using ruptures, use below hyperparameters
    model = "l2"
    pen_beta = 0.0001
    min_size = 5
    # end of ruptures hyperparameters

    param_window_size = 10
    trade_window_size = 1
    K = 0.001
    param_method = 'compound'
    HJB_M = 800
    HJB_N = 800
    alpha = 0.95
    ### end of hyperparameters

    asset_name = "S&P500"
    # asset_name = "Bitcoin"
    # asset_name = "Etherium"
    # asset_name = "BNB"
    asset_ticker_dict = {'S&P500': '^GSPC', 'Bitcoin': 'BTC-USD', 'Etherium': 'ETH-USD', 'BNB': 'BNB-USD'}
    asset_ticker = asset_ticker_dict[asset_name]
    rho_ticker = "^TNX"

    # get price history for given ticker in given period
    price_df = DataDownload.fetch_price_data(ticker=asset_ticker, start_date=start_date, end_date=end_date, force_download=True)
    rho_df = DataDownload.fetch_price_data(ticker=rho_ticker, start_date=start_date, end_date=end_date, force_download=True)
    rho_df = rho_df.rename(columns={'Adj Close': 'rho'})
    data_df = price_df[['Adj Close']].copy()
    data_df = data_df.rename(columns={'Adj Close': 'Close'})
    data_df['AssetReturnsCC'] = data_df['Close'].pct_change().dropna()  # Return Close on Close
    data_df = data_df.merge(rho_df['rho'], how='left', left_index=True, right_index=True, sort=False)
    data_df = data_df.fillna(method='pad')

    # Identify bull and bear regimes
    # regime_df = ParameterEstimation.identify_regimes(data_df=data_df, threshold_list=thresholds)
    regime_df = ParameterEstimation.identify_regimes_cpd(data_df=data_df, window_size=trade_window_size, model=model,
                                                         pen_beta=pen_beta, min_size=min_size)
    print()
    print(regime_df)

    # Plot
    fig_1 = ParameterEstimation.plot_regimes(data_df=data_df, regime_df=regime_df, path=base_folder)

    # compute parameters based on identified regimes and given window length in years
    params_df = ParameterEstimation.compute_window_parameters(data_df=data_df, regime_df=regime_df, k=K,
                                                              param_window_size=param_window_size,
                                                              trade_window_size=trade_window_size, method=param_method,
                                                              alpha=alpha)
    params_df.to_excel(base_folder+"parameters_config_{:.1f}.xlsx".format(version))
    print()
    print(params_df.head())

    # run backtest
    strategy_df = run_backtest(data_df=data_df, params_df=params_df, hjb_m=HJB_M, hjb_n=HJB_N)
    print()
    print(strategy_df.head())

    end_time = time()
    minutes = int((end_time - start_time) // 60)
    seconds = (end_time - start_time) % 60
    print()
    time_str = "Total Run Time = {0:d} min and {1:.2f} sec".format(minutes, seconds)
    print(time_str)

    res_df = PortfolioAnalytics.evaluate_portfolio_performance(asset_wise_pf_ret=strategy_df['StrategyDailyReturn'],
                                                               pf_cum_ret=strategy_df['StrategyCumReturn'],
                                                               benchmark_ret=strategy_df['AssetReturnsCC'],
                                                               benchmark_cum_ret=strategy_df['BenchmarkCumReturn'],
                                                               positions=strategy_df['Position'], benchmark_name=asset_name)

    config_desc = """
        start_date = {start_date} \n
        end_date = {end_date} \n
        threshold_list = {threshold_list}
        param_window_size = {param_window_size} \n
        trade_window_size = {trade_window_size} \n
        param_method = {param_method} \n
        K = {K} \n
        HJB_M = {m} \n
        HJB_N = {n} \n
        alpha = {alpha} \n
        *Change Log*
        First regime is taken from start of regime even if it is before the window start.
        Last regime is skipped as it is not over.
        Added sigma in EWA list.
        Taking duration weighted mean for mu1 and mu2 for P(t).
        Taking simple mean for mu1 and mu2 for HJB solver.
        Using change point detection for regime identification
        Change point detection:
        min_size = {min_size}
        *** \n
        {time_str} \n\n\n
        ---------------
        Results Summary 
        ---------------\n\n{res_df}
        """.format(start_date=start_date.date(), end_date=end_date.date(), threshold_list=thresholds,
                   param_window_size=param_window_size, trade_window_size=trade_window_size, param_method=param_method,
                   K=K, m=HJB_M, n=HJB_N, time_str=time_str, res_df=res_df.to_string(), alpha=alpha, min_size=min_size)

    # Plot
    plot_strategy_performance(strategy_df=strategy_df, path=base_folder, output_suffix=output_suffix, fig_1=fig_1, config_desc=config_desc)

    print('end')
