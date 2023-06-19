import pandas as pd
from pandas.tseries.offsets import BDay, BMonthBegin
import numpy as np
import matplotlib.pyplot as plt
import pickle
import yfinance as yf


def calc_sharpe(returns):
    """
    Sharpe ratio for given returns.
    """
    return (returns.mean() / returns.std()) * np.sqrt(252)


def calc_cagr(cum_returns):
    """
    Compounded Annual Growth Rate for given cumulative returns.
    """
    return ((cum_returns[-1]) ** (252 / len(cum_returns))) - 1


def calc_ann_vol(returns):
    """
    Volatility for given returns.
    """
    return returns.std()*np.sqrt(252)


def calc_pf_beta(returns, bm_returns):
    return (np.cov(returns, bm_returns)[0][1] / np.var(bm_returns))

def calc_pf_alpha(returns, bm_returns):
    return (returns.mean() - calc_pf_beta(returns, bm_returns) * bm_returns.mean()) * 252


def calc_max_drawdown(cum_returns):
    return min(-1 + cum_returns / cum_returns.cummax())


def calc_mdd_start_end(cum_returns):
    dd = -1 + cum_returns / cum_returns.cummax()
    mdd = min(dd)
    mdd_end = dd[dd == mdd].index[0]
    mdd_start = cum_returns[cum_returns.index <= mdd_end].idxmax()
    return mdd_start.date(), mdd_end.date()


def prep_str(val, mul=1, unit='', rnd=None):
    """
    Prepare string formatting for nice printing of results.
    """
    if rnd is not None:
        return '{0}{1}'.format(round(val*mul, rnd), unit)
    else:
        return '{0}{1}'.format(val*mul, unit)


def evaluate_portfolio_performance(asset_wise_pf_ret, pf_cum_ret, benchmark_ret, benchmark_cum_ret, positions,
                                   benchmark_name='S&P 500', plot=False, plot_title='Performance', save_fig=False, save_file_path='./Performance.jpg'):
    """
    Evaluate portfolio performance, compare to benchmark, create performance table, make plot.
    :param benchmark_name: string, name of the benchmark
    :param plot: boolean, whether to plot the strategy vs benchmark cumulative returns
    :param plot_title: string, title for plot if plot=True
    :param save_fig: boolean, whether to save the plot
    :return: tuple of (performance summary dataframe, strategy cumulative returns, benchmark cumulative returns) if ret_only_cum_ret and ret_only_sharpe are False
    """

    if type(asset_wise_pf_ret) == pd.DataFrame:
        pf_ret = asset_wise_pf_ret.sum(axis=1)
    else:
        pf_ret = asset_wise_pf_ret

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

    # Max Drawdown
    pf_max_dd = prep_str(calc_max_drawdown(pf_cum_ret), mul=100, unit='%', rnd=2)
    bm_max_dd = prep_str(calc_max_drawdown(benchmark_cum_ret), mul=100, unit='%', rnd=2)

    # Max Drawdown start and end date
    pf_mdd_start, pf_mdd_end = calc_mdd_start_end(pf_cum_ret)
    bm_mdd_start, bm_mdd_end = calc_mdd_start_end(benchmark_cum_ret)

    # Beta
    beta = prep_str(calc_pf_beta(pf_ret, benchmark_ret), rnd=2)

    # Alpha
    alpha = prep_str(calc_pf_alpha(pf_ret, benchmark_ret), mul=100, unit='%', rnd=2)

    # trade count
    pf_trade_count = np.abs((positions - positions.shift(1)).dropna()).sum().sum()

    # performance table
    res_df = pd.DataFrame({'Strategy': [pf_val, pf_cagr, pf_vol, pf_sharpe, pf_max_dd, (pf_mdd_start, pf_mdd_end), beta, alpha, pf_trade_count],
                           benchmark_name: [bm_val, bm_cagr, bm_vol, bm_sharpe, bm_max_dd, (bm_mdd_start, bm_mdd_end), '1', '-', '-']},
                          index=['Cumulative Return', 'CAGR', 'Annualized Volatility', 'Sharpe', 'Max Drawdown', 'Max Drawdown Period', 'Beta', "Jensen's Alpha", 'Trade Count'])

    return res_df