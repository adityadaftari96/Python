"""
Author: Aditya Daftari
"""
# import sys
# sys.path.append("../")

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
from Misc import PortfolioAnalytics

from bokeh.plotting import figure, show
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import LinearAxis, Range1d, Span
from bokeh.models.layouts import Tabs, TabPanel
from bokeh.models.widgets import PreText


def p_t1(pt, params, S0, S1):
    dt = 1 / 252

    g_pt = -pt * (params['lambda1'] + params['lambda2']) + params['lambda2'] - (
            pt * (params['mu1'] - params['mu2']) * (1 - pt) * (
            pt * (params['mu1'] - params['mu2']) + params['mu2'] - 0.5 * (params['sigma'] ** 2))) / (
                   params['sigma'] ** 2)

    return min(max(pt + g_pt * dt + (pt * (1 - pt) * (params['mu1'] - params['mu2']) * np.log(S1 / S0)) / (params['sigma'] ** 2), 0), 1)


def run_backtest(data_df, params_df, hjb_m, hjb_n):
    thresholds_df = params_df.copy()
    thresholds_df['P_b'] = 1
    thresholds_df['P_s'] = 1

    # compute buy and sell thresholds for each trading window
    print('computing thresholds for period:')
    for i, row in params_df.iterrows():
        print(row['TradeWindowStart'].date(), row['TradeWindowEnd'].date())
        thresholds_df.loc[i, 'P_b'], thresholds_df.loc[i, 'P_s'] = HJB_Solver.get_thresholds(params=row.to_dict(), m=hjb_m, n=hjb_n)

    # prepare daily thresholds
    trade_df = thresholds_df.copy()
    trade_df = trade_df.set_index(['TradeWindowStart']).reindex(data_df.index)
    trade_df = trade_df.fillna(method='pad').dropna()
    trade_df = trade_df.merge(data_df[['Close', 'AssetReturnsCC']], how='left', left_index=True, right_index=True, sort=False)

    # compute daily conditional probability of being in bull market, and daily positions accordingly
    trade_df['P(t)'] = np.NaN
    trade_df['Position'] = np.NaN

    trade_df.loc[trade_df.index[0], 'P(t)'] = (trade_df.loc[trade_df.index[0], 'P_b'] + trade_df.loc[trade_df.index[0], 'P_s']) / 2.0
    trade_df.loc[trade_df.index[0], 'Position'] = 0

    for i in range(1, len(trade_df.iloc[1:])):
        p_t0 = trade_df.loc[trade_df.index[i-1], 'P(t)']
        s0 = trade_df.loc[trade_df.index[i-1], 'Close']
        s1 = trade_df.loc[trade_df.index[i], 'Close']

        trade_df.loc[trade_df.index[i], 'P(t)'] = p_t1(pt=p_t0, params=trade_df.loc[trade_df.index[i], :].to_dict(), S0=s0, S1=s1)

        if trade_df.loc[trade_df.index[i], 'P(t)'] > trade_df.loc[trade_df.index[i], 'P_b']:
            trade_df.loc[trade_df.index[i], 'Position'] = 1
        elif trade_df.loc[trade_df.index[i], 'P(t)'] < trade_df.loc[trade_df.index[i], 'P_s']:
            trade_df.loc[trade_df.index[i], 'Position'] = 0

    trade_df['Position'] = trade_df['Position'].fillna(method='pad')  # position taken at end of day

    # creating results df
    strategy_df = trade_df.copy()
    strategy_df = strategy_df.filter(['P_b', 'P_s', 'P(t)', 'Close', 'AssetReturnsCC', 'Position'])
    strategy_df = strategy_df.merge(data_df[['rho']], how='left', left_index=True, right_index=True, sort=False)  # live rho

    strategy_df['Position'] = strategy_df['Position'].shift(1)  # position available at begin of day
    strategy_df['rho'] = strategy_df['rho'].shift(1)  # on any day, risk-free rate earned will be as of previous close
    strategy_df = strategy_df.iloc[1:]  # first day is ignored as it is part of the training period

    strategy_df['rho'] = ((1 + (strategy_df['rho'] / 100.0)) ** (1.0 / 252.0)) - 1
    strategy_df['StrategyDailyReturn'] = np.where(strategy_df['Position'] == 1, strategy_df['AssetReturnsCC'], strategy_df['rho'])
    strategy_df['StrategyCumReturn'] = np.cumprod(1 + strategy_df['StrategyDailyReturn'])
    strategy_df['BenchmarkCumReturn'] = strategy_df['Close'] / (strategy_df['Close'][0])
    return strategy_df


def plot_strategy_performance(strategy_df, path, output_suffix, fig_1, config_desc):

    mask = strategy_df['Position'] == 1
    long_only_ret = strategy_df['StrategyCumReturn'].copy()
    risk_free_ret = strategy_df['StrategyCumReturn'].copy()
    long_only_ret[~mask] = np.NaN
    risk_free_ret[mask] = np.NaN

    output_file(filename=path + "Results_{0}.html".format(output_suffix))

    # Create a new plot
    fig_2 = figure(width=1500, height=800, x_axis_type='datetime', y_range=(-0.2, 1.2))

    source = ColumnDataSource(data={
        'date': np.array(strategy_df.index, dtype=np.datetime64),
        'P_b': strategy_df['P_b'].values,
        'P_s': strategy_df['P_s'].values,
        'P(t)': strategy_df['P(t)'].values,
        'BenchmarkCumReturn': strategy_df['BenchmarkCumReturn'].values,
        'StrategyCumReturn': strategy_df['StrategyCumReturn'].values,
        'Strategy(Long)': long_only_ret.values,
        'Strategy(Flat)': risk_free_ret.values,
    })

    # Primary y-axis
    fig_2.line(x='date', y='P_b', color="cyan", line_width=1, source=source, legend_label="P_b")
    fig_2.line(x='date', y='P_s', color="pink", line_width=1, source=source, legend_label="P_s")
    fig_2.line(x='date', y='P(t)', color="brown", line_width=1, source=source, legend_label="P(t)")
    fig_2.xaxis.axis_label = 'Date'
    fig_2.yaxis.axis_label = 'Bull Market Probability and Thresholds'

    # secondary y-axis
    y_min = min(min(strategy_df['BenchmarkCumReturn'].values), min(strategy_df['StrategyCumReturn'].values)) - 0.2
    y_max = max(max(strategy_df['BenchmarkCumReturn'].values), max(strategy_df['StrategyCumReturn'].values)) + 0.2
    fig_2.extra_y_ranges['secondary'] = Range1d(y_min, y_max)
    fig_2.add_layout(LinearAxis(y_range_name="secondary", axis_label='Cumulative Returns'), 'right')

    fig_2.line(x='date', y='BenchmarkCumReturn', color="orange", line_width=1, source=source, legend_label="Benchmark Return", y_range_name="secondary")
    strat = fig_2.line(x='date', y='StrategyCumReturn', color="grey", line_width=1, source=source, legend_label="Strategy Cumulative Return", y_range_name="secondary")
    fig_2.line(x='date', y='Strategy(Flat)', color="red", line_width=1, source=source, legend_label="Strategy Cumulative Return", y_range_name="secondary")
    fig_2.line(x='date', y='Strategy(Long)', color="green", line_width=1, source=source, legend_label="Strategy Cumulative Return", y_range_name="secondary")

    # add vertical line on plot to follow mouse
    # vline = Span(dimension='height', line_color='black', line_dash='dashed', line_width=2)
    # fig_2.add_layout(vline)

    # Add the hover tool with the specified tooltips and mode='vline'
    fig_2.add_tools(HoverTool(
        tooltips=[
            ('date', '@date{%F}'),
            ('P_b', '@P_b{0.00 a}'),
            ('P_s', '@P_s{0.00 a}'),
            ('P(t)', '@{P(t)}{0.00 a}'),
            ('BenchmarkCumReturn', '@BenchmarkCumReturn{0.00 a}'),
            ('StrategyCumReturn', '@StrategyCumReturn{0.00 a}'),
        ],

        formatters={
            '@date': 'datetime',  # use 'datetime' formatter for '@date' field
            # use default 'numeral' formatter for other fields
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline',
        renderers=[strat]
    ))
    fig_2.legend.click_policy = "hide"

    # show(fig_2)

    config_txt = PreText(text=config_desc)

    # Create the panels for each plot
    panel1 = TabPanel(child=fig_1, title="Regimes")
    panel2 = TabPanel(child=fig_2, title="Strategy results")
    panel3 = TabPanel(child=config_txt, title="Config")

    # Create the tabs
    tabs = Tabs(tabs=[panel3, panel1, panel2])

    # Show the tabs layout
    save(tabs)
    # save(fig_2)


########################################################################################################################
if __name__ == "__main__":

    print('start')

    base_folder = "/Users/adityadaftari/Library/CloudStorage/OneDrive-nyu.edu/Documents/NYU MFE/DeepAlpha/results/"
    version = 2.3
    output_suffix = "config_{:.1f}".format(version)

    # Period for parameter estimation with start and end date included
    start_date = datetime(1962, 1, 3)
    # end_date = datetime(2011, 12, 31)
    end_date = datetime(2023, 5, 30)

    # hyperparameters
    thresh_bull = 0.2
    thresh_bear = 0.2
    param_window_size = 10
    trade_window_size = 1
    K = 0.001
    param_method = 'compound'
    HJB_M = 800
    HJB_N = 4000

    start_time = time()
    # get price history for given ticker in given period
    asset_ticker = "^GSPC"
    rho_ticker = "^TNX"
    price_df = DataDownload.fetch_price_data(ticker=asset_ticker, start_date=start_date, end_date=end_date, force_download=True)
    rho_df = DataDownload.fetch_price_data(ticker=rho_ticker, start_date=start_date, end_date=end_date, force_download=True)
    rho_df = rho_df.rename(columns={'Adj Close': 'rho'})
    data_df = price_df[['Adj Close']].copy()
    data_df = data_df.rename(columns={'Adj Close': 'Close'})
    data_df['AssetReturnsCC'] = data_df['Close'].pct_change().dropna()  # Return Close on Close
    data_df = data_df.merge(rho_df['rho'], how='left', left_index=True, right_index=True, sort=False)
    data_df = data_df.fillna(method='pad')

    # Identify bull and bear regimes
    regime_df = ParameterEstimation.identify_regimes(data_df=data_df, thresh_bull=thresh_bull, thresh_bear=thresh_bear)
    print()
    print(regime_df)

    # Plot
    fig_1 = ParameterEstimation.plot_regimes(data_df=data_df, regime_df=regime_df, path=base_folder)

    # compute parameters based on identified regimes and given window length in years
    params_df = ParameterEstimation.compute_window_parameters(data_df=data_df, regime_df=regime_df, k=K,
                                                              param_window_size=param_window_size,
                                                              trade_window_size=trade_window_size, method=param_method)
    print()
    print(params_df.head())

    # run backtest
    strategy_df = run_backtest(data_df=data_df, params_df=params_df, hjb_m=HJB_M, hjb_n=HJB_M)
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
                                                               positions=strategy_df['Position'])

    config_desc = """
        start_date = {start_date} \n
        end_date = {end_date} \n
        thresh_bull = {thresh_bull} \n
        thresh_bear = {thresh_bear} \n
        param_window_size = {param_window_size} \n
        trade_window_size = {trade_window_size} \n
        param_method = {param_method} \n
        K = {K} \n
        HJB_M = {m} \n
        HJB_N = {n} \n
        {time_str} \n\n\n
        ---------------
        Results Summary 
        ---------------\n\n{res_df}
        """.format(start_date=start_date.date(), end_date=end_date.date(), thresh_bull=thresh_bull, thresh_bear=thresh_bear,
                   param_window_size=param_window_size, trade_window_size=trade_window_size, param_method=param_method,
                   K=K, m=HJB_M, n=HJB_N, time_str=time_str, res_df=res_df.to_string())

    # Plot
    plot_strategy_performance(strategy_df=strategy_df, path=base_folder, output_suffix=output_suffix, fig_1=fig_1, config_desc=config_desc)

    print('end')









