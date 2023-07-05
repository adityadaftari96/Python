"""
This file goes through the full history of the underlying asset price and finds bull and bear markets
assuming a regime switching model. Two thresholds are assumed as hyperparameters, thresh_bull to identify bull periods,
and thresh_bear to identify bear periods. Bull periods are those where cumulative return from last trough has exceeded
thresh_bull. Bull periods last till the next peak after which a bear period is identified. Bear periods are those where
cumulative returns from last peak drop more than thresh_bear. Bear periods last till the next trough after which another
bull period is identified.
"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from datetime import datetime
import yfinance as yf
from time import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import DataDownload
from dateutil.relativedelta import relativedelta
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import LinearAxis, Range1d, Span


def calibrate_regime_thresholds_EQ(period, threshold_list):
    columns = ['Date', 'BearThreshold', 'BullThreshold']
    reg_thresh_df = pd.DataFrame(threshold_list, columns=columns)
    reg_thresh_df = reg_thresh_df.set_index(['Date'])
    index = pd.bdate_range(start=reg_thresh_df.index[0], end=period[1])
    reg_thresh_df = reg_thresh_df.reindex(index)
    reg_thresh_df = reg_thresh_df.fillna(method='ffill')
    reg_thresh_df = reg_thresh_df.loc[reg_thresh_df.index >= period[0]]
    return reg_thresh_df


def calibrate_regime_thresholds_Crypto(period, threshold_list, asset_prices):
    columns = ['Date', 'BearThreshold', 'BullThreshold']
    reg_thresh_df = pd.DataFrame(threshold_list, columns=columns)
    reg_thresh_df = reg_thresh_df.set_index(['Date'])
    reg_thresh_df = reg_thresh_df.reindex(asset_prices.index)
    reg_thresh_df = reg_thresh_df.fillna(method='ffill')
    reg_thresh_df = reg_thresh_df.loc[reg_thresh_df.index >= period[0]]
    return reg_thresh_df


def identify_regimes(data_df, threshold_list):

    close_prices = data_df['Close']
    returns = data_df['AssetReturnsCC'].dropna()

    regime_switch_dates_list = list()

    # cum_ret_curr = 1  # to track current regime cumulative return
    cum_ret_next = 1  # to track next regime cumulative return

    peak = close_prices[0]
    trough = peak

    peak_date = close_prices.index[0]
    trough_date = peak_date

    bear = False
    bull = False

    regime_thresholds_df = calibrate_regime_thresholds_EQ(period=(returns.index[0], returns.index[-1]), threshold_list=threshold_list)
    # regime_thresholds_df = calibrate_regime_thresholds_Crypto(period=(returns.index[0], returns.index[-1]), threshold_list=threshold_list, asset_prices=close_prices)

    for date in returns.index:
        cum_ret_next = cum_ret_next * (1 + returns[returns.index == date][0])
        close_price = close_prices[close_prices.index == date][0]

        thresh_bull = (1 + regime_thresholds_df.loc[date, 'BullThreshold'])
        thresh_bear = (1 - regime_thresholds_df.loc[date, 'BearThreshold'])

        if cum_ret_next <= thresh_bear:
            if bull or bear == bull:
                regime_switch_dates_list.append(peak_date)
            bear = True
            bull = False
            trough_date = date
            trough = close_price
            cum_ret_next = 1  # reset next regime cumulative return if trough or peak is found
        elif cum_ret_next >= thresh_bull:
            if bear or bear == bull:
                regime_switch_dates_list.append(trough_date)
            bear = False
            bull = True
            peak_date = date
            peak = close_price
            cum_ret_next = 1  # reset next regime cumulative return if trough or peak is found

        if bear and close_price < trough:  # new trough
            trough_date = date
            trough = close_price
            cum_ret_next = 1  # reset next regime cumulative return if trough or peak is found
        elif bull and close_price > peak:  # new peak
            peak_date = date
            peak = close_price
            cum_ret_next = 1  # reset next regime cumulative return if trough or peak is found

    if bear:
        regime_switch_dates_list.append(trough_date)
        if date != trough_date:
            regime_switch_dates_list.append(date)

    if bull:
        regime_switch_dates_list.append(peak_date)
        if date != peak_date:
            regime_switch_dates_list.append(date)

    regime_df = pd.DataFrame(data=regime_switch_dates_list, columns=['Top/Bottom'])
    regime_df['TB_shifted'] = regime_df['Top/Bottom'].shift(1)
    regime_df['Index'] = regime_df.apply(lambda x: close_prices.loc[x['Top/Bottom']], axis=1)
    regime_df['% Move'] = regime_df['Index'].pct_change()
    regime_df['Regime'] = np.where(regime_df['% Move'] > 0, 1, 0)
    regime_df['Mean'] = regime_df.apply(lambda x: returns[(returns.index > x['TB_shifted']) & (returns.index <= x['Top/Bottom'])].mean(), axis=1)
    regime_df['Stdev'] = regime_df.apply(lambda x: returns[(returns.index > x['TB_shifted']) & (returns.index <= x['Top/Bottom'])].std(), axis=1)
    regime_df['Duration'] = regime_df.apply(lambda x: returns[(returns.index > x['TB_shifted']) & (returns.index <= x['Top/Bottom'])].count(), axis=1)
    regime_df = regime_df.drop(columns=['TB_shifted'])
    return regime_df


def plot_regimes(data_df, regime_df, path):
    close_prices = data_df['Close']
    bull_periods = regime_df.loc[regime_df['% Move'] > 0, 'Top/Bottom']
    bear_periods = regime_df.loc[regime_df['% Move'] < 0, 'Top/Bottom']

    # output_file(filename=path + "regimes.html", title="Regimes")

    # create a new plot with a specific size
    fig = figure(width=1500, height=800, x_axis_type='datetime')

    source = ColumnDataSource(data={
        'date': np.array(close_prices.index, dtype=np.datetime64),
        'index': close_prices.values,
        'returns': 100*data_df['AssetReturnsCC'].values,
    })

    fig.xaxis.axis_label = 'Date'
    fig.yaxis.axis_label = 'Index'

    # secondary y-axis
    y_min = 100*min(data_df['AssetReturnsCC'].dropna().values)
    y_max = 100*max(data_df['AssetReturnsCC'].dropna().values)
    fig.extra_y_ranges['secondary'] = Range1d(y_min, y_max)
    fig.add_layout(LinearAxis(y_range_name="secondary", axis_label='Daily Returns'), 'right')

    # add a renderer
    index = fig.line(x='date', y='index', line_width=1, color='orange', source=source, legend_label="Index Level")
    fig.line(x='date', y='returns', line_width=1, color='blue', source=source, legend_label="Daily Return (%)", y_range_name="secondary")
    fig.add_tools(HoverTool(
        tooltips=[
            ('date', '@date{%F}'),
            ('index', '@index{0.2f}'),  # use @{ } for field names with spaces
            ('returns', '@returns{0.4f} %'),  # use @{ } for field names with spaces
        ],

        formatters={
            '@date': 'datetime',  # use 'datetime' formatter for '@date' field
            # '@{index}': 'printf',  # use 'printf' formatter for '@{adj close}' field
            # use default 'numeral' formatter for other fields
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='vline',
        renderers=[index]
    ))

    # Define the y-coordinates for the start and end points of the lines
    y_start = [min(close_prices)] * len(bull_periods)
    y_end = [max(close_prices)] * len(bull_periods)

    # Add the vertical lines to the plot using the segment glyph
    fig.segment(bull_periods, y_start, bull_periods, y_end, line_color="green", line_width=1, line_dash="dashed")

    y_start = [min(close_prices)] * len(bear_periods)
    y_end = [max(close_prices)] * len(bear_periods)

    # Add the vertical lines to the plot using the segment glyph
    fig.segment(bear_periods, y_start, bear_periods, y_end, line_color="red", line_width=1, line_dash="dashed")

    fig.legend.click_policy = "hide"

    # save the results to a file
    # show(fig)
    # save(fig)
    return fig


def compute_window_parameters(data_df, regime_df, k, param_window_size, trade_window_size, alpha, method='average'):
    close_prices = data_df['Close']
    close_prices = close_prices.reindex(pd.bdate_range(start=close_prices.index[0], end=close_prices.index[-1]))
    close_prices = close_prices.fillna(method='pad')

    returns = data_df['AssetReturnsCC']

    regime_start = regime_df['Top/Bottom'].iloc[0]
    regime_end = regime_df['Top/Bottom'].iloc[-1]

    window_start = regime_start
    window_end = min(window_start + relativedelta(years=param_window_size), regime_end)

    params_old = dict()
    params_df = pd.DataFrame()
    while window_end < regime_end:
        regime_window_df = regime_df.loc[(regime_df['Top/Bottom'] >= window_start) & (regime_df['Top/Bottom'] <= window_end), :]
        regime_window_df = regime_window_df.reset_index(drop=True)
        regime_window_df = regime_window_df.rename(columns={'Top/Bottom': 'WindowEnd'})
        regime_window_df['WindowStart'] = regime_window_df['WindowEnd'].shift(1)  # start of trend
        regime_window_df = regime_window_df.filter(['WindowStart', 'WindowEnd', 'Regime'])
        if regime_window_df.empty:
            # handle no recorded regimes within current window. This would happen if recorded regime start and end date
            # is before and after current window start and end date.
            regime_window_df['WindowStart'] = [window_start]
            regime_window_df['WindowEnd'] = window_end
            temp_prices = close_prices.loc[(close_prices.index >= window_start) & (close_prices.index <= window_end)]
            perc_move = -1 + (temp_prices.iloc[-1] / temp_prices.iloc[0])
            regime_window_df['Regime'] = 1 if perc_move > 0 else 0
        elif regime_window_df['WindowEnd'].iloc[0] > window_start:
            # add another regime from start of current window to the first recorded regime
            first_window_start = regime_df.loc[regime_df['Top/Bottom'] <= regime_window_df['WindowEnd'].iloc[0]]['Top/Bottom'].iloc[-2]
            regime_window_df.loc[0, 'WindowStart'] = first_window_start
            temp_prices = close_prices.loc[(close_prices.index >= window_start) & (close_prices.index <= regime_window_df['WindowEnd'].iloc[0])]
            perc_move = -1 + (temp_prices.iloc[-1] / temp_prices.iloc[0])
            regime_window_df.loc[0, 'Regime'] = 1 if perc_move > 0 else 0
        else:
            # if first row window end is same as window start, that means the first regime is recorded in the second row.
            # So delete first row.
            regime_window_df = regime_window_df.iloc[1:].reset_index(drop=True)

        # commenting below code to not include last regime in parameter estimation window
        # if regime_window_df['WindowEnd'].iloc[-1] < window_end:
            # regime = abs(regime_window_df['Regime'].iloc[-1] - 1)  # opposite of last regime in regime_df
            # temp_dict = {'WindowStart': [regime_window_df['WindowEnd'].iloc[-1]], 'WindowEnd': [window_end], 'Regime': [regime]}
            # temp_df = pd.DataFrame.from_dict(temp_dict)
            # regime_window_df = pd.concat([regime_window_df, temp_df], axis=0)

        regime_window_df['Mean'] = regime_window_df.apply(lambda x: returns[(returns.index > x['WindowStart']) & (returns.index <= x['WindowEnd'])].mean(), axis=1)
        regime_window_df['Duration'] = regime_window_df.apply(lambda x: returns[(returns.index > x['WindowStart']) & (returns.index <= x['WindowEnd'])].count(), axis=1)
        regime_window_df['Stdev'] = regime_window_df.apply(lambda x: returns[(returns.index > x['WindowStart']) & (returns.index <= x['WindowEnd'])].std(), axis=1)
        regime_window_df['StartIndex'] = regime_window_df.apply(lambda x: close_prices.loc[BDay(1).rollback(x['WindowStart'])], axis=1)
        regime_window_df['EndIndex'] = regime_window_df.apply(lambda x: close_prices.loc[BDay(1).rollback(x['WindowEnd'])], axis=1)
        regime_window_df['% Move'] = -1 + (regime_window_df['EndIndex'] / regime_window_df['StartIndex'])

        params = dict()
        params['ParamWindowStart'] = window_start
        params['ParamWindowEnd'] = window_end
        params['TradeWindowStart'] = window_end
        params['TradeWindowEnd'] = min(params['TradeWindowStart'] + relativedelta(years=trade_window_size), regime_end)
        params['rho'] = (np.median(data_df.loc[(data_df.index >= params['ParamWindowStart']) & (data_df.index <= params['ParamWindowEnd']), 'rho'])) / 100.0

        bull_df = regime_window_df.loc[regime_window_df['Regime'] == 1, :]
        bear_df = regime_window_df.loc[regime_window_df['Regime'] == 0, :]

        params['sigma1'] = bull_df['Stdev'].mean() * np.sqrt(252) if not bull_df.empty else 0.0
        params['sigma2'] = bear_df['Stdev'].mean() * np.sqrt(252) if not bear_df.empty else 0.0
        params['lambda1'] = (252 / bull_df['Duration']).mean()
        params['lambda2'] = (252 / bear_df['Duration']).mean()

        if method == 'average':
            # takes average of daily mean return of all bull periods and then annualizes to get mu1. Similar calculation for mu2.
            # sigma = average of sigma1 and sigma2
            params['mu1'] = bull_df['Mean'].mean() * 252
            params['mu2'] = bear_df['Mean'].mean() * 252
            if np.isnan(params['sigma1']):
                params['sigma'] = params['sigma2']
            elif np.isnan(params['sigma2']):
                params['sigma'] = params['sigma1']
            else:
                params['sigma'] = (params['sigma1'] + params['sigma2']) / 2
        elif method == 'compound':
            # takes % move for each bull period, annualizes it and takes average to get mu1. Similar calculation for mu2.
            # sigma is computed as annualized std dev of all daily returns in current window
            params['HJBmu1'] = (((1 + bull_df['% Move']) ** (252 / bull_df['Duration'])) - 1).mean()
            params['HJBmu2'] = (((1 + bear_df['% Move']) ** (252 / bear_df['Duration'])) - 1).mean()
            params['mu1'] = ((((1 + bull_df['% Move']) ** (252 / bull_df['Duration'])) - 1) * bull_df['Duration']).sum() / bull_df['Duration'].sum() if len(bull_df) > 0 else 0
            params['mu2'] = ((((1 + bear_df['% Move']) ** (252 / bear_df['Duration'])) - 1) * bear_df['Duration']).sum() / bear_df['Duration'].sum() if len(bear_df) > 0 else 0
            params['sigma'] = returns[(returns.index > window_start) & (returns.index <= window_end)].std() * np.sqrt(252)

        # NaN check: Replace NaN with 0, then take EWA so effective value will be (1-alpha)*(old value).
        # use exponential weighting to update parameters
        if not params_old:
            params_old = params
        update_params_list = ['mu1', 'mu2', 'lambda1', 'lambda2', 'sigma', 'HJBmu1', 'HJBmu2']
        # N = 5
        # alpha = 2 / N
        for key in update_params_list:
            params[key] = params[key] if not np.isnan(params[key]) else 0.0
            params[key] = ((1 - alpha) * params_old[key]) + (alpha * params[key])
        params_old = params

        params['ParamWindowStart'] = [params['ParamWindowStart']]  # making it a list for conversion to df from dict
        temp_df = pd.DataFrame.from_dict(params)
        params_df = pd.concat([params_df, temp_df], axis=0)

        window_start = window_start + relativedelta(years=trade_window_size)
        window_end = window_start + relativedelta(years=param_window_size)

    params_df = params_df.reset_index(drop=True)
    params_df['trade_window_size'] = trade_window_size
    params_df['K'] = k

    return params_df


if __name__ == "__main__":
    print('start')
    start_time = time()
    base_folder = "/Users/adityadaftari/Library/CloudStorage/OneDrive-nyu.edu/Documents/NYU MFE/DeepAlpha/results/config1/"

    # Period for parameter estimation with start and end date included
    start_date = datetime(1962, 1, 3)
    end_date = datetime(2009, 3, 9)

    # Thresholds for bull and bear period
    thresh_bull = 0.24
    thresh_bear = 0.19
    param_window_size = 10
    trade_window_size = 1
    K = 0.001

    # get price history for given ticker in given period
    ticker = "^GSPC"
    rho_ticker = "^TNX"
    price_df = DataDownload.fetch_price_data(ticker=ticker, start_date=start_date, end_date=end_date)
    rho_df = DataDownload.fetch_price_data(ticker=rho_ticker, start_date=start_date, end_date=end_date)
    rho_df = rho_df.rename(columns={'Adj Close': 'rho'})
    data_df = price_df[['Adj Close']].copy()
    data_df = data_df.rename(columns={'Adj Close': 'Close'})
    data_df['AssetReturnsCC'] = data_df['Close'].pct_change().dropna()  # Return Close on Close
    data_df = data_df.merge(rho_df['rho'], how='left', left_index=True, right_index=True, sort=False)
    data_df = data_df.fillna(method='pad')

    # Identify bull and bear regimes
    regime_df = identify_regimes(data_df=data_df, thresh_bull=thresh_bull, thresh_bear=thresh_bear)
    print()
    print(regime_df)

    end_time = time()
    minutes = int((end_time - start_time) // 60)
    seconds = (end_time - start_time) % 60
    print()
    print("Time Taken for computation of regimes = {0:d} min and {1:.2f} sec".format(minutes, seconds))

    # Plot
    plot_regimes(data_df=data_df, regime_df=regime_df, path=base_folder)

    # compute parameters based on identified regimes and given window length in years
    params_df = compute_window_parameters(data_df=data_df, regime_df=regime_df, k=K,
                                          param_window_size=param_window_size, trade_window_size=trade_window_size,
                                          alpha=0.95)
    print()
    print(params_df)

    print('end')
