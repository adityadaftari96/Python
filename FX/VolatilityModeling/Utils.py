"""
https://quant.stackexchange.com/questions/44300/mixed-local-stochastic-volatility-model-in-quantlib
https://www.quantconnect.com/learning/articles/introduction-to-options/local-volatility-and-stochastic-volatility
"""

import warnings
warnings.filterwarnings('ignore')

import QuantLib as ql
import numpy as np
import pandas as pd
import itertools

from scipy.stats import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time


# Some utility functions used later to plot 3D vol surfaces, generate paths, and generate vol surface from Heston params
def plot_vol_surface(vol_surface, plot_years=np.arange(0.1, 3, 0.1), plot_strikes=np.arange(70, 130, 1), funct='blackVol', title=''):
    if type(vol_surface) != list:
        surfaces = [vol_surface]
    else:
        surfaces = vol_surface

    fig = plt.figure()
    plt.title(title)
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(plot_strikes, plot_years)

    for surface in surfaces:
        method_to_call = getattr(surface, funct)

        Z = np.array([method_to_call(float(y), float(x))
                      for xr, yr in zip(X, Y)
                      for x, y in zip(xr, yr)]
                     ).reshape(len(X), len(X[0]))

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.1)

    fig.colorbar(surf, shrink=0.5, aspect=5)


def generate_multi_paths_df(sequence, num_paths):
    spot_paths = []
    vol_paths = []

    for i in range(num_paths):
        sample_path = sequence.next()
        values = sample_path.value()

        spot, vol = values

        spot_paths.append([x for x in spot])
        vol_paths.append([x for x in vol])

    df_spot = pd.DataFrame(spot_paths, columns=[spot.time(x) for x in range(len(spot))])
    df_vol = pd.DataFrame(vol_paths, columns=[spot.time(x) for x in range(len(spot))])

    return df_spot, df_vol


def create_vol_surface_mesh_from_heston_params(today, calendar, spot, v0, kappa, theta, rho, sigma,
                                               rates_curve_handle, dividend_curve_handle,
                                               strikes = np.linspace(40, 200, 161), tenors = np.linspace(0.1, 3, 60)):
    quote = ql.QuoteHandle(ql.SimpleQuote(spot))

    heston_process = ql.HestonProcess(rates_curve_handle, dividend_curve_handle, quote, v0, kappa, theta, sigma, rho)
    heston_model = ql.HestonModel(heston_process)
    heston_handle = ql.HestonModelHandle(heston_model)
    heston_vol_surface = ql.HestonBlackVolSurface(heston_handle)

    data = []
    for strike in strikes:
        data.append([heston_vol_surface.blackVol(tenor, strike) for tenor in tenors])

    expiration_dates = [calendar.advance(today, ql.Period(int(365*t), ql.Days)) for t in tenors]
    implied_vols = ql.Matrix(data)
    feller = 2 * kappa * theta - sigma ** 2

    return expiration_dates, strikes, implied_vols, feller


def get_maturity_list(current_date, tenor_list, calendar):
    maturity_list = []
    for tenor in tenor_list:
        maturity = calendar.advance(current_date, ql.Period(tenor))
        maturity_list.append(maturity)
    return maturity_list


def convert_vols(vol_df):
    cp_vol_df = vol_df[['ATM']]
    cp_vol_df['25dC'] = vol_df['ATM'] + vol_df['25D BF'] + 0.5 * vol_df['25D RR']
    cp_vol_df['25dP'] = vol_df['ATM'] + vol_df['25D BF'] - 0.5 * vol_df['25D RR']
    cp_vol_df['10dC'] = vol_df['ATM'] + vol_df['10D BF'] + 0.5 * vol_df['10D RR']
    cp_vol_df['10dP'] = vol_df['ATM'] + vol_df['10D BF'] - 0.5 * vol_df['10D RR']
    return cp_vol_df


def solve_for_strike(spot, ttm, sigma, r_dom, r_for, delta, opt_type):
    k = 0
    d1 = norm.ppf(delta * np.exp(r_for * ttm))
    if opt_type == 'C':  # Call option
        d1 *= -1
    k = spot * np.exp(d1 * sigma * np.sqrt(ttm) + (r_dom - r_for + 0.5 * (sigma ** 2)) * ttm)
    return k


def create_imp_vol_surface(spot, cp_vol_df, dates, day_count, current_date, calendar, r_dom, r_for):
    strike_set = {spot}
    pillars = ['10dP', '25dP', 'ATM', '25dC', '10dC']
    cp_vol_df = cp_vol_df.filter(pillars)
    row = -1
    strikes = []
    for maturity in dates:
        ttm = day_count.yearFraction(current_date, maturity)
        row += 1
        vol_smile = cp_vol_df.iloc[row]
        mat_strikes = []
        for m in pillars:
            if m == 'ATM':
                k = spot
            else:
                sigma = vol_smile[m]
                delta = float(m[:2]) / 100
                opt_type = m[-1]
                k = round(solve_for_strike(spot, ttm, sigma, r_dom, r_for, delta, opt_type), 4)
                strike_set.add(k)
            mat_strikes.append(k)
        strikes.append(mat_strikes)

    strike_list = sorted(list(strike_set))
    strike_list = strike_list[:-3]

    vol_matrix = []
    for i in range(0, len(dates)):
        vol_surface = ql.BlackVolTermStructureHandle(ql.BlackVarianceSurface(current_date, calendar, [dates[i]], strikes[i],
                                                                             cp_vol_df.iloc[i].values.reshape(-1, 1).tolist(),
                                                                             day_count))
        vol_matrix.append([vol_surface.blackVol(dates[i], j, True) for j in strike_list])
    vol_array = np.array(vol_matrix).transpose()

    matrix = []
    for i in range(0, vol_array.shape[0]):
        matrix.append(vol_array[i].tolist())

    vol_surface = ql.Matrix(matrix)

    return strike_list, vol_surface, vol_matrix


def setup_model(_dom_ts, _for_ts, _spot, init_condition=(0.02, 0.2, 0.5, 0.1, 0.01)):
    """
    Initializes the HestonModel and the AnalyticHestonEngine prior to calibration
    """
    theta, kappa, sigma, rho, v0 = init_condition
    spot_quote = ql.QuoteHandle(ql.SimpleQuote(_spot))
    process = ql.HestonProcess(_dom_ts, _for_ts, spot_quote, v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    return model, engine


def setup_helpers(engine, expiration_dates, strikes, data, ref_date, spot, dom_ts, for_ts, calendar):
    """
    Construct the Heston model helpers and returns an array of these objects
    """
    heston_helpers = []
    grid_data = []
    for i, date in enumerate(expiration_dates):
        for j, s in enumerate(strikes):
            t = calendar.businessDaysBetween(ref_date, date)
            p = ql.Period(t, ql.Days)
            vol = data[i][j]
            vol_quote = ql.QuoteHandle(ql.SimpleQuote(vol))
            helper = ql.HestonModelHelper(p, calendar, spot, s, vol_quote, dom_ts, for_ts, ql.BlackCalibrationHelper.PriceError)
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)
            grid_data.append((date, s))
    return heston_helpers, grid_data


def cost_function_generator(model, helpers, norm=False):
    """
    Set the cost function to be used by the Scipy modules
    """
    def cost_function(params):
        params_ = ql.Array(list(params))
        model.setParams(params_)
        error = [h.calibrationError() for h in helpers]
        if norm:
            return np.sqrt(np.sum(np.abs(error)))
        else:
            return error

    return cost_function


def calibration_report(helpers, grid_data, spot, detailed=False):
    """
    Evaluate the quality of the fit
    """
    avg = 0.0
    if detailed:
        print("%15s %25s %15s %15s %20s" % ("Strikes", "Expiry", "Market Value", "Model Value", "Error per $100 underlying"))
        print("=" * 100)
    for i, opt in enumerate(helpers):
        err = 100 * (abs(opt.modelValue() - opt.marketValue()) / spot)
        date, strike = grid_data[i]
        if detailed:
            print("%15.2f %25s %14.5f %15.5f %20.7f " % (strike, str(date), opt.marketValue(), opt.modelValue(), err))
        avg += err
    avg = avg / len(helpers)
    if detailed:
        print("-" * 100)
    summary_str = "Mean Error per $100 underlying : %5.9f" % avg
    print(summary_str)
    return avg


def print_time(start_time, end_time):
    minutes = int((end_time - start_time) // 60)
    seconds = (end_time - start_time) % 60
    time_str = "Model Run Time = {0:d} min and {1:.2f} sec".format(minutes, seconds)
    print(time_str)


def option_price_slv_mc(option_type, strike, dom_ts, for_ts, spot_quote, v0, kappa, theta, sigma, rho, ttm, timestep,
                        leverage_functon, mixing_factor=0.6, seed=14, num_paths=25000, barrier=None, expiry=-1):
    """
    Price european vanilla and exotic options using stochastic local vol model and monte carlo simulations
    """
    # Create a path generator and generate paths from the Stochastic Local Vol process
    start = time()
    times = ql.TimeGrid(ttm, timestep)

    heston_process = ql.HestonProcess(dom_ts, for_ts, spot_quote, v0, kappa, theta, sigma, rho)
    stoch_local_process = ql.HestonSLVProcess(heston_process, leverage_functon, mixing_factor)
    dimension = stoch_local_process.factors()

    rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(dimension * timestep, ql.UniformRandomGenerator(seed=seed)))
    seq = ql.GaussianMultiPathGenerator(stoch_local_process, times, rng, False)

    df_spot, df_vol = generate_multi_paths_df(seq, num_paths)

    prem = 0
    if option_type == 'VP':
        # Vanilla Put
        prem = (strike - df_spot.iloc[:, expiry]).clip(lower=0).mean()
    elif option_type == 'DAOP':
        # Down and Out Put
        ko_paths = (df_spot.iloc[:, :timestep+2+expiry] < barrier).sum(axis=1) > 0
        payoffs = (strike - df_spot.iloc[:, expiry]).clip(lower=0)
        payoffs[ko_paths] = 0
        prem = payoffs.mean()

    end = time()
    print_time(start, end)

    return prem





# dupire_local_vol_handle = ql.LocalVolTermStructureHandle(dupire_local_vol)
# tGrid, xGrid = 2000, 200
# bsm_process = ql.BlackScholesMertonProcess(spot_quote, for_ts, dom_ts, dupire_local_vol_handle)
# engine = ql.FdBlackScholesVanillaEngine(bsm_process, tGrid, xGrid, localVol=True)
# option_type = ql.Option.Put
# payoff = ql.PlainVanillaPayoff(option_type, strike)
# europeanExercise = ql.EuropeanExercise(maturity_date)
# europeanOption = ql.VanillaOption(payoff, europeanExercise)
# europeanOption.setPricingEngine(engine)
# option_value_lv = europeanOption.NPV()




