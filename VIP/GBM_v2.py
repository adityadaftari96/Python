import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import DataDownload
from datetime import datetime

def get_sim_paths(ticker, start_date, end_date, num):

    #--------------------------------------------------- GEOMETRIC BROWNIAN MOTION ------------------------------------------------

    # Parameter Definitions

    # So    :   initial stock price
    # dt    :   time increment -> a day in our case
    # T     :   length of the prediction time horizon(how many time points to predict, same unit with dt(days))
    # N     :   number of time points in prediction the time horizon -> T/dt
    # t     :   array for time points in the prediction time horizon [1, 2, 3, .. , N]
    # mu    :   mean of historical daily returns
    # sigma :   standard deviation of historical daily returns
    # b     :   array for brownian increments
    # W     :   array for brownian path

    data = DataDownload.fetch_price_data(ticker=ticker, start_date=start_date, end_date=end_date, force_download=True)
    base_data = data[['Adj Close']].copy()
    base_data = base_data.rename(columns={'Adj Close': 'Close'})

    returns = base_data['Close'].pct_change()
    #print(returns.tolist())

    # Parameter Assignments
    So = base_data['Close'].head(1).values
    print(So)
    dt = 1 # day   # User input
    n_of_wkdays = len(base_data) - 1
    T = n_of_wkdays # days  # User input -> follows from pred_end_date
    N = T / dt
    t = np.arange(1, int(N) + 1)
    mu = np.mean(returns)
    sigma = np.std(returns)
    scen_size = num # User input
    b = {str(scen): np.random.normal(0, 1, int(N)) for scen in range(1, scen_size + 1)}
    W = {str(scen): b[str(scen)].cumsum() for scen in range(1, scen_size + 1)}

    # Calculating drift and diffusion components
    drift = (mu - 0.5 * sigma**2) * t
    #print(drift)
    diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, scen_size + 1)}
    #print(diffusion)

    # Making the predictions
    S = np.array([So * np.exp(drift + diffusion[str(scen)]) for scen in range(1, scen_size + 1)]) 

    # add So to the beginning series
    S = np.hstack((np.array([So for scen in range(scen_size)]), S)) 

    Preds_df = pd.DataFrame(S.swapaxes(0, 1)).set_index(base_data.index).reset_index(drop = False)
    Preds_df = Preds_df.set_index('Date')
    #Preds_df.columns = str(Preds_df.columns)
    return Preds_df

    