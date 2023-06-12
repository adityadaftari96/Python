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


def fetch_price_data(ticker, start_date, end_date, force_download=False):
    path = "./data/{}.pkl".format(ticker.replace("^", ""))

    if not force_download:
        try:
            with open(path, "rb") as handle:
                df = pickle.load(handle)
            df = df.loc[(df.index >= start_date) & (df.index <= end_date), :]
        except FileNotFoundError:
            df = yf.download(tickers=[ticker], start=start_date, end=end_date + BDay(1))
            with open(path, 'wb') as handle:
                pickle.dump(df, handle)
    else:
        df = yf.download(tickers=[ticker], start=start_date, end=end_date + BDay(1))
        with open(path, 'wb') as handle:
            pickle.dump(df, handle)

    return df


def identify_regimes(price_df, thresh_bull, thresh_bear):
    thresh_bull = (1 + thresh_bull)
    thresh_bear = (1 - thresh_bear)

    close_prices = price_df['Adj Close']
    returns = close_prices.pct_change().dropna()  # Return Close on Close

    regime_switch_dates_list = list()

    # cum_ret_curr = 1  # to track current regime cumulative return
    cum_ret_next = 1  # to track next regime cumulative return

    peak = close_prices[0]
    trough = peak

    peak_date = close_prices.index[0]
    trough_date = peak_date

    bear = False
    bull = False

    for date in returns.index:
        # cum_ret_curr = cum_ret_curr * (1 + returns[returns.index == date][0])
        cum_ret_next = cum_ret_next * (1 + returns[returns.index == date][0])
        close_price = close_prices[close_prices.index == date][0]

        if cum_ret_next <= thresh_bear:
            bear = True
            bull = False
            regime_switch_dates_list.append(peak_date)
            trough_date = date
            trough = close_price
            cum_ret_next = 1  # reset next regime cumulative return if trough or peak is found
        elif cum_ret_next >= thresh_bull:
            bear = False
            bull = True
            regime_switch_dates_list.append(trough_date)
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
    regime_df['Mean'] = regime_df.apply(lambda x: returns[(returns.index > x['TB_shifted']) & (returns.index <= x['Top/Bottom'])].mean(), axis=1)
    regime_df['Stdev'] = regime_df.apply(lambda x: returns[(returns.index > x['TB_shifted']) & (returns.index <= x['Top/Bottom'])].std(), axis=1)
    regime_df['Duration'] = regime_df.apply(lambda x: returns[(returns.index > x['TB_shifted']) & (returns.index <= x['Top/Bottom'])].count(), axis=1)
    regime_df = regime_df.drop(columns=['TB_shifted'])
    return regime_df


def plot_regimes(price_df, regime_df):
    close_prices = price_df['Adj Close']
    bull_periods = regime_df.loc[regime_df['% Move'] > 0, 'Top/Bottom']
    bear_periods = regime_df.loc[regime_df['% Move'] < 0, 'Top/Bottom']
    fig = plt.figure(figsize=(12, 8))
    plt.plot(close_prices)
    plt.vlines(x=bull_periods, colors='green', ls='--', ymin=[min(close_prices)]*len(bull_periods), ymax=[max(close_prices)]*len(bull_periods))
    plt.vlines(x=bear_periods, colors='red', ls='--', ymin=[min(close_prices)]*len(bear_periods), ymax=[max(close_prices)]*len(bear_periods))
    plt.title('Index level and regimes')
    plt.xlabel('Time')
    plt.ylabel('Index')
    plt.show()


def compute_window_parameters(regime_df, price_df, window):
    close_prices = price_df['Adj Close']
    returns = close_prices.pct_change().dropna()  # Return Close on Close

    params = dict()
    params['mu1'] = regime_df.loc[regime_df['% Move'] > 0, 'Mean'].mean() * 252
    params['mu2'] = regime_df.loc[regime_df['% Move'] < 0, 'Mean'].mean() * 252
    params['lambda1'] = (252/regime_df.loc[regime_df['% Move'] > 0, 'Duration']).mean()
    params['lambda2'] = (252/regime_df.loc[regime_df['% Move'] < 0, 'Duration']).mean()
    params['sigma1'] = regime_df.loc[regime_df['% Move'] > 0, 'Stdev'].mean() * np.sqrt(252)
    params['sigma2'] = regime_df.loc[regime_df['% Move'] < 0, 'Stdev'].mean() * np.sqrt(252)
    # params['sigma'] = regime_df.apply(lambda x: returns[(returns.index > x['TB_shifted']) & (returns.index <= x['Top/Bottom'])].mean(), axis=1)
    return params


if __name__ == "__main__":
    print('start')
    start_time = time()

    # Period for parameter estimation with start and end date included
    start_date = datetime(1962, 1, 3)
    end_date = datetime(2009, 3, 9)

    # Thresholds for bull and bear period
    thresh_bull = 0.24
    thresh_bear = 0.19

    # get price history for given ticker in given period
    ticker = "^GSPC"
    price_df = fetch_price_data(ticker=ticker, start_date=start_date, end_date=end_date)

    # Identify bull and bear regimes
    regime_df = identify_regimes(price_df=price_df, thresh_bull=thresh_bull, thresh_bear=thresh_bear)
    print()
    print(regime_df)

    end_time = time()
    minutes = int((end_time - start_time) // 60)
    seconds = (end_time - start_time) % 60
    print()
    print("Time Taken for computation of regimes = {0:d} min and {1:.2f} sec".format(minutes, seconds))

    # Plot
    plot_regimes(price_df=price_df, regime_df=regime_df)

    # compute parameters based on identified regimes and given window length in years
    params = compute_window_parameters(regime_df=regime_df, price_df=price_df, window=10)
    print()
    print(params)

    print('end')
