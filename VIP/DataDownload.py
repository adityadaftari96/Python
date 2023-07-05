import yfinance as yf
import pickle
from pandas.tseries.offsets import BDay


def fetch_price_data(ticker, start_date, end_date, force_download=False):
    path = "./data/{}.pkl".format(ticker.replace("^", ""))

    if not force_download:
        try:
            with open(path, "rb") as handle:
                df = pickle.load(handle)
            df = df.loc[(df.index >= start_date) & (df.index <= end_date), :]
        except FileNotFoundError:
            df = yf.download(tickers=ticker, start=start_date, end=end_date + BDay(1))
            with open(path, 'wb') as handle:
                pickle.dump(df, handle)
    else:
        df = yf.download(tickers=ticker, start=start_date, end=end_date + BDay(1))
        with open(path, 'wb') as handle:
            pickle.dump(df, handle)

    return df