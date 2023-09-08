from Utils import *


if __name__ == "__main__":
    # World State for Vanilla Pricing
    spot = 100
    rate = 0.0
    today = ql.Date(1, 7, 2020)
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()

    # Set up the flat risk-free curves
    riskFreeCurve = ql.FlatForward(today, rate, ql.Actual365Fixed())

    flat_ts = ql.YieldTermStructureHandle(riskFreeCurve)
    dividend_ts = ql.YieldTermStructureHandle(riskFreeCurve)

    # create and plot a vol surface using some random params in a heston process
    dates, strikes, vols, feller = create_vol_surface_mesh_from_heston_params(today, calendar, spot, 0.0225, 1.0, 0.0625, -0.25, 0.3, flat_ts, dividend_ts)

    local_vol_surface = ql.BlackVarianceSurface(today, calendar, dates, strikes, vols, day_count)

    # Plot the vol surface ...
    plot_vol_surface(local_vol_surface)

    # Calculate the Dupire instantaneous vol
    spot_quote = ql.QuoteHandle(ql.SimpleQuote(spot))

    local_vol_surface.setInterpolation("bicubic")
    local_vol_handle = ql.BlackVolTermStructureHandle(local_vol_surface)
    local_vol = ql.LocalVolSurface(local_vol_handle, flat_ts, dividend_ts, spot_quote)
    local_vol.enableExtrapolation()

    # Plot the Dupire surface ...
    plot_vol_surface(local_vol, funct='localVol')

    # Calibrate a Heston process
    # Create new heston model
    v0 = 0.015
    kappa = 2.0
    theta = 0.065
    rho = -0.3
    sigma = 0.45
    spot = 1007
    feller = 2 * kappa * theta - sigma ** 2

    heston_process = ql.HestonProcess(flat_ts, dividend_ts, spot_quote, v0, kappa, theta, sigma, rho)
    heston_model = ql.HestonModel(heston_process)

    # How does the vol surface look at the moment?
    heston_handle = ql.HestonModelHandle(heston_model)
    heston_vol_surface = ql.HestonBlackVolSurface(heston_handle)

    # Plot the vol surface ...
    plot_vol_surface([local_vol_surface, heston_vol_surface])

    plt.show()