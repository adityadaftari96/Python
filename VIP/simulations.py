import pandas as pd
import numpy as np
from datetime import datetime

import GBM_v2
import sim_strat
import DataDownload

if __name__ == "__main__":
      # Period for parameter estimation with start and end date included
    start_date = datetime(1962, 1, 3)
    # end_date = datetime(2011, 12, 31)
    end_date = datetime(2023, 5, 30)
    ticker = "^GSPC"
    num = 20 #number of simulations
    
    df = GBM_v2.get_sim_paths(ticker, start_date, end_date, num)
    rho_df = DataDownload.fetch_price_data(ticker='^TNX', start_date=start_date, end_date=end_date, force_download=True)
    rho_df = rho_df.rename(columns={'Adj Close': 'rho'})
    #print(df)

    all_res = []
    for i in range(num):
        path = pd.DataFrame()
        path['Close'] = df[i]
        results = sim_strat.run_strat(path, rho_df)
        # print('results')
        # print(results)

    