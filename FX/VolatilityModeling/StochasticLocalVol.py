import QuantLib

from Utils import *
from time import time
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution


if __name__ == "__main__":

    # region Settings

    # World State for Vanilla Pricing
    # spot = 100
    spot = 1.091  # as of Aug 15 EOD
    r_dom = 0.0
    r_for = 0.0
    calculation_date = ql.Date(15, 8, 2023)
    ql.Settings.instance().evaluationDate = calculation_date
    vol_file = "./data/The Quarry  - Vol Data.xlsx"
    sheet_name = "EURUSD vol surface"
    cal_dom = ql.UnitedStates(ql.UnitedStates.FederalReserve)
    cal_for = ql.TARGET()
    calendar = ql.JointCalendar(cal_dom, cal_for)
    day_count = ql.Actual365Fixed()
    #endregion

    # region Set up the flat risk-free curves
    dom_curve = ql.FlatForward(calculation_date, r_dom, ql.Actual365Fixed())
    dom_ts = ql.YieldTermStructureHandle(dom_curve)
    for_curve = ql.FlatForward(calculation_date, r_for, ql.Actual365Fixed())
    for_ts = ql.YieldTermStructureHandle(for_curve)
    #endregion

    # region import implied vol data, create and plot implied vol surface

    # dates, strikes, vols, feller = create_vol_surface_mesh_from_heston_params(today, calendar, spot, 0.0225, 1.0, 0.0625, -0.25, 0.3, dom_ts, for_ts)
    vol_df = pd.read_excel(vol_file, sheet_name=sheet_name)
    vol_df = vol_df.drop([0]).set_index(['Unnamed: 0'])
    vol_df = vol_df / 100
    cp_vol_df = convert_vols(vol_df)
    dates = get_maturity_list(calculation_date, vol_df.index.tolist(), calendar)
    strikes, vols, data = create_imp_vol_surface(spot, cp_vol_df, dates, day_count, calculation_date, calendar, r_dom, r_for)
    implied_vol_surface = ql.BlackVarianceSurface(calculation_date, calendar, dates, strikes, vols, day_count)

    # Plot the vol surface ...
    plot_vol_surface(implied_vol_surface, plot_years=np.arange(0.1, 10, 0.1), plot_strikes=np.arange(strikes[0], strikes[-1], 0.01), funct='blackVol', title='Imp Vol')
    #endregion

    # region Calculate the Dupire instantaneous vol
    spot_quote = ql.QuoteHandle(ql.SimpleQuote(spot))
    implied_vol_surface.setInterpolation("bilinear")
    implied_vol_handle = ql.BlackVolTermStructureHandle(implied_vol_surface)
    dupire_local_vol = ql.LocalVolSurface(implied_vol_handle, dom_ts, for_ts, spot_quote)
    dupire_local_vol.enableExtrapolation()
    # ToDo: create your own Dupire local vol surface using Dupire formula, and compare with local_vol surface from quantlib.

    # Plot the Dupire surface ...
    plot_vol_surface(dupire_local_vol, plot_years=np.arange(0.1, 10, 0.1), plot_strikes=np.arange(strikes[0], strikes[-1], 0.01), funct='localVol', title='Dupire Local Vol')
    #endregion

    # region Calibrate a Heston process

    # Initialize new heston model with dummy parameters
    # order of params - theta, kappa, sigma, rho, v0
    init_params_1 = (0.10, 5.0, 0.9, -0.5, 0.08)
    init_params_2 = (0.1, 0.1, 0.1, 0.1, 0.1)
    bounds = [(0, 1), (0.01, 15), (0.01, 1.), (-1, 1), (0, 1.0)]
    summary = []

    ## Scipy LS Model 1
    heston_model1, engine1 = setup_model(dom_ts, for_ts, spot, init_condition=init_params_1)
    heston_helpers1, grid_data1 = setup_helpers(engine1, dates, strikes, data, calculation_date, spot, dom_ts, for_ts, calendar)
    print("Init Params = ", list(heston_model1.params()))

    start_time = time()
    cost_function = cost_function_generator(heston_model1, heston_helpers1)
    sol = least_squares(cost_function, np.array(init_params_1), verbose=True)
    theta, kappa, sigma, rho, v0 = sol.x
    print("theta = %f, kappa = %f, sigma = %f, rho = %f, v0 = %f" % (theta, kappa, sigma, rho, v0))
    end_time = time()
    print_time(start_time, end_time)
    error = calibration_report(heston_helpers1, grid_data1, spot, detailed=True)
    summary.append(["Scipy LS1", error] + list(heston_model1.params()))

    # Plot the calibrated vol surface
    heston_handle = ql.HestonModelHandle(heston_model1)
    heston_vol_surface = ql.HestonBlackVolSurface(heston_handle)
    plot_vol_surface([implied_vol_surface, heston_vol_surface], plot_years=np.arange(0.1, 10, 0.1), plot_strikes=np.arange(strikes[0], strikes[-1], 0.01), title='Heston Stoch Vol LS Model 1')

    results = pd.DataFrame(summary, columns=["Name", "Avg Abs Error", "Theta", "Kappa", "Sigma", "Rho", "V0"], index=[''] * len(summary))
    print(results)
    #endregion

    # region Run the local vol fitting and calculate the leverage function
    # Calibrate via Monte-Carlo
    maturity_date = ql.Date(15, 8, 2024)
    generator_factory = ql.MTBrownianGeneratorFactory(43)

    # calibration_paths_vars = [2 ** 15, 2 ** 17, 2 ** 19, 2 ** 20]
    calibration_paths_vars = [2 ** 15]
    time_steps_per_year, n_bins = 365, 201

    # for calibration_paths in calibration_paths_vars:
    calibration_paths = calibration_paths_vars[0]
    print("Paths: {}".format(calibration_paths))
    stoch_local_mc_model = ql.HestonSLVMCModel(dupire_local_vol, heston_model1, generator_factory, maturity_date, time_steps_per_year, n_bins, calibration_paths)

    a = time()
    leverage_functon = stoch_local_mc_model.leverageFunction()
    b = time()

    print("calibration took {0:2.1f} seconds".format(b - a))
    plot_vol_surface(leverage_functon, funct='localVol', plot_years=np.arange(0.1, 0.98, 0.1), plot_strikes=np.arange(strikes[0], strikes[-1], 0.01), title='Leverage Function')

    # plt.show()
    #endregion

    # region price European Vanilla OTM put option by local vol and stochastic local vol

    date, strike = grid_data1[800]
    ttm = 1.0
    timestep = 365
    dT = ttm / timestep
    theta, kappa, sigma, rho, v0 = heston_model1.params()

    option_value_mkt = heston_helpers1[800].marketValue()
    option_value_slv = option_price_slv_mc('VP', strike, dom_ts, for_ts, spot_quote, v0, kappa, theta, sigma, rho,
                                           ttm, timestep, leverage_functon)
    print("Vanilla Put Market value = {:.4}, slv mc value = {:.4}".format(option_value_mkt, option_value_slv))

    #endregion

    # region price European Down and Out Barrier OTM put option by local vol and stochastic local vol
    barrier = 0.9
    barr_option_value_slv = option_price_slv_mc('DAOP', strike, dom_ts, for_ts, spot_quote, v0, kappa, theta, sigma, rho,
                                                ttm, timestep, leverage_functon, barrier=barrier)

    ## greeks using slv mc

    # delta
    dS = 0.01
    barr_option_value_slv_sp = option_price_slv_mc('DAOP', strike, dom_ts, for_ts, ql.QuoteHandle(ql.SimpleQuote(spot + dS)),
                                                   v0, kappa, theta, sigma, rho, ttm, timestep, leverage_functon, barrier=barrier)
    barr_option_value_slv_sn = option_price_slv_mc('DAOP', strike, dom_ts, for_ts, ql.QuoteHandle(ql.SimpleQuote(spot - dS)),
                                                   v0, kappa, theta, sigma, rho, ttm, timestep, leverage_functon, barrier=barrier)
    delta = (barr_option_value_slv_sp - barr_option_value_slv_sn) / (2 * dS)

    # gamma
    gamma = (barr_option_value_slv_sp + barr_option_value_slv_sn - 2 * barr_option_value_slv) / (dS ** 2)

    # vega
    dV = 0.01
    barr_option_value_slv_vp = option_price_slv_mc('DAOP', strike, dom_ts, for_ts, spot_quote, v0,
                                                   kappa, theta + dV, sigma, rho, ttm, timestep, leverage_functon, barrier=barrier)
    barr_option_value_slv_vn = option_price_slv_mc('DAOP', strike, dom_ts, for_ts, spot_quote, v0,
                                                   kappa, theta - dV, sigma, rho, ttm, timestep, leverage_functon, barrier=barrier)
    vega = (barr_option_value_slv_vp - barr_option_value_slv_vn) / (2 * dV)

    # theta
    barr_option_value_slv_tn = option_price_slv_mc('DAOP', strike, dom_ts, for_ts, spot_quote, v0, kappa, theta, sigma, rho,
                                                   ttm, timestep, leverage_functon, barrier=barrier, expiry=-10)
    theta = - (barr_option_value_slv - barr_option_value_slv_tn) / (10*dT)

    print("DAOP option value = {:.4}, delta = {:.4}, gamma = {:.4}, vega = {:.4}, theta = {:.4}".format(barr_option_value_slv, delta, gamma, vega, theta))

    #endregion


