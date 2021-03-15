from itertools import chain
import pandas as pd
import numpy as np
import datetime
import time
import os
from pathlib import Path
from alpha_vantage.timeseries import TimeSeries
import sys
from tqdm import tqdm

ALPHA_VANTAGE_DIR_PATH = Path("alphadata").absolute()
# GET TICKERS
tickers = os.listdir(ALPHA_VANTAGE_DIR_PATH)

slippage = .005 # 0.5% slippage per trade
dict_dfs = dict()

def generate_monthly_stats(df):
    #print(df)
    log_return = df["Close"].apply(np.log).diff()
    half_way_point = len(df) // 2

    return {
        "Open": df["Open"].iloc[0],
        "High": df["High"].max(),
        "Low": df["Low"].min(),
        "Close": df["Close"].iloc[-1],
        "Volume": df["Volume"].sum(),
        "first_half_log_return_mean": log_return.iloc[:half_way_point].mean(),
        "first_half_log_return_std": log_return.iloc[:half_way_point].std(),
        "second_half_log_return_mean": log_return.iloc[half_way_point:].mean(),
        "second_half_log_return_std": log_return.iloc[half_way_point:].std(),
        "first_second_half_log_return_diff": (
            log_return.iloc[half_way_point:].sum()
            - log_return.iloc[:half_way_point].sum()
        ),
        "log_return_mean": log_return.mean(),
        "log_return_std": log_return.std(),
        "log_return_min": log_return.min(),
        "log_return_max": log_return.max(),
        "month_log_return": np.log(df["Close"].iloc[-1] / df["Open"].iloc[0]),
        "pct_bull": (log_return > 0).mean()
    }

s = datetime.datetime.now()

for t in tqdm(tickers):
    # t = tickers[0]
    try:
        temp = (
                pd.read_csv(ALPHA_VANTAGE_DIR_PATH / f"{t}", index_col=0, parse_dates=True)
                .groupby(pd.Grouper(freq="1M"))
                .apply(generate_monthly_stats)
            )


        df = pd.DataFrame()
        for i in range(len(temp)):
            df = df.append(pd.DataFrame.from_dict(temp.iloc[i],orient='index').T, ignore_index=True)
        # df = df.append(pd.DataFrame.from_dict(temp.iloc[1],orient='index').T, ignore_index=True)
        df.set_index = temp.index

        df.index = list(temp.index)

        df["next_month_log_return"] = np.log(
                np.exp(df["month_log_return"].shift(-1)) * (1 - slippage) / (1 + slippage)
            )
        dict_dfs[t] = df
    except:
        print(f"Skipping: {t}")

e = datetime.datetime.now()

print(e-s)

dict_dfs[t].to_csv(t)

for ticker in tqdm(dict_dfs):
    dict_dfs[ticker].to_csv(f'tickers_summary/{ticker}')
