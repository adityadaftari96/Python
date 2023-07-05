import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from dateutil.relativedelta import relativedelta
from datetime import datetime
from time import time

import HJB_Solver
import DataDownload
import ParameterEstimation
import PortfolioAnalytics
import Trend_Following_Trading

from bokeh.plotting import figure, show
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import LinearAxis, Range1d, Span
from bokeh.models.layouts import Tabs, TabPanel
from bokeh.models.widgets import PreText

def run_strat(df, rho_df) :

    #print('start')

    #base_folder = "./results/"
    #version = 2.3
    #output_suffix = "config_{:.1f}".format(version)

    # Period for parameter estimation with start and end date included
    start_date = df.head(1).index[0]
    # end_date = datetime(2011, 12, 31)
    end_date = df.tail(1).index[0]

    # hyperparameters
    thresholds = [
        # start date, bear threshold, bull threshold
        [start_date, 0.20, 0.20],
        # [start_date, 0.60, 0.60],
        # [datetime(2017, 11, 9), 0.6, 0.6],
        [datetime(2010, 1, 1), 0.18, 0.20],
        # [datetime(2020, 1, 1), 0.17, 0.23],
    ]
    param_window_size = 10
    trade_window_size = 1
    K = 0.001
    param_method = 'compound'
    HJB_M = 100
    HJB_N = 100
    alpha = 0.95

    start_time = time()
    # get price history for given ticker in given period
    #asset_ticker = "^GSPC"
    #price_df = DataDownload.fetch_price_data(ticker=asset_ticker, start_date=start_date, end_date=end_date, force_download=True)
    #data_df = price_df[['Adj Close']].copy()
    data_df = df
    data_df['AssetReturnsCC'] = data_df['Close'].pct_change().dropna()  # Return Close on Close
    data_df = data_df.merge(rho_df['rho'], how='left', left_index=True, right_index=True, sort=False)
    data_df = data_df.fillna(method='pad')
    data_df.index = pd.to_datetime(data_df.index)

    # Identify bull and bear regimes
    regime_df = ParameterEstimation.identify_regimes(data_df=data_df, threshold_list=thresholds)
    #print()
    #print(regime_df)

    # Plot
    #fig_1 = ParameterEstimation.plot_regimes(data_df=data_df, regime_df=regime_df, path=base_folder)

    # compute parameters based on identified regimes and given window length in years
    params_df = ParameterEstimation.compute_window_parameters(data_df=data_df, regime_df=regime_df, k=K,
                                                              param_window_size=param_window_size,
                                                              trade_window_size=trade_window_size, method=param_method, alpha=alpha)
    #print()
    #print(params_df.head())

    # run backtest
    strategy_df = Trend_Following_Trading.run_backtest(data_df=data_df, params_df=params_df, hjb_m=HJB_M, hjb_n=HJB_N)
    #print()
    #print(strategy_df.head())

    end_time = time()
    minutes = int((end_time - start_time) // 60)
    seconds = (end_time - start_time) % 60
    print()
    time_str = "Total Run Time = {0:d} min and {1:.2f} sec".format(minutes, seconds)
    #print(time_str)

    res_df = PortfolioAnalytics.evaluate_portfolio_performance(asset_wise_pf_ret=strategy_df['StrategyDailyReturn'],
                                                               pf_cum_ret=strategy_df['StrategyCumReturn'],
                                                               benchmark_ret=strategy_df['AssetReturnsCC'],
                                                               benchmark_cum_ret=strategy_df['BenchmarkCumReturn'],
                                                               positions=strategy_df['Position'])

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
           Taking duration weighted mean for mu1 and mu2 for P(t) instead of simple average.
           Taking simple mean for mu1 and mu2 for HJB solver instead of simple average.
           *** \n
           {time_str} \n\n\n
           ---------------
           Results Summary 
           ---------------\n\n{res_df}
           """.format(start_date=start_date.date(), end_date=end_date.date(), threshold_list=thresholds,
                      param_window_size=param_window_size, trade_window_size=trade_window_size,
                      param_method=param_method,
                      K=K, m=HJB_M, n=HJB_N, time_str=time_str, res_df=res_df.to_string(), alpha=alpha)

    # Plot
    #plot_strategy_performance(strategy_df=strategy_df, path=base_folder, output_suffix=output_suffix, fig_1=fig_1, config_desc=config_desc)

    #print('end')
    return res_df
