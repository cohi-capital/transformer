"""
Prepare KuCoin OHLCV data for binary classification, where targets are cases
with 10% price jump or higher.

We separate data in test and validation and put them into numpy arrays on a
per symbol basis.

Floats are 32-bit. Data is stored in pickle files.
"""
import os
import pickle
import numpy as np
import pandas as pd

from datetime import datetime

MARKET_DATA_FILTERED_CCXT_KUCOIN_20230812 = 'market_data_filtered_kucoin_ccxt_only_20230101_20230812.parquet.nosync'
TRAIN_DATA = 'train.pkl'
VALIDATION_DATA = 'val.pkl'
N_SPLITS = 10
BLOCK_SIZE = 168  # 24 * 7, 1 week


def generate_pkl():
    # Add bitcoin information to each row
    df_base = pd.read_parquet(MARKET_DATA_FILTERED_CCXT_KUCOIN_20230812)

    btc = df_base.loc[df_base['identifier'] == 'kucoin-BTC-USDT'].copy()
    btc.columns = 'btc_' + btc.columns.values

    df_merged = df_base.merge(btc, how="left", left_on="time_open", right_on="btc_time_open")

    df_merged.drop(
        columns=[
            'exchange',
            'market_symbol',
            'base_currency',
            'base_currency_name',
            'quote_currency',
            'volume_usd',
            'btc_time_open',
            'btc_exchange',
            'btc_market_symbol',
            'btc_base_currency',
            'btc_base_currency_name',
            'btc_quote_currency',
            'btc_volume_usd',
            'btc_identifier'
        ],
        inplace=True
    )

    # Get cut-off time between test and validation
    min_date = pd.Timestamp(df_merged['time_open'].min(), unit='ms')
    max_date = pd.Timestamp(df_merged['time_open'].max(), unit='ms')
    time_between = max_date - min_date + pd.Timedelta(hours=1)
    cutoff_time = min_date + (time_between * (N_SPLITS - 1) / N_SPLITS)

    # Go through symbol by symbol to generate training and validation data
    dict_train = dict()
    dict_validation = dict()

    total_symbols = len(df_merged.identifier.unique())
    start_time = datetime.now()
    train_size = []
    val_size = []

    for i, identifier in enumerate(df_merged.identifier.unique()):
        if i % 100 == 0:
            print(f'{datetime.now()}| Elapsed: {datetime.now() - start_time} | '
                  f'Processed {i} of {total_symbols} symbols. | '
                  f'Train size avg: {np.mean(train_size):.2f} | Test size avg: {np.mean(val_size):.2f}')

        tmp = df_merged.loc[df_merged['identifier'] == identifier].copy()
        tmp.sort_values(['time_open'], inplace=True)

        # Convert unix timestamp back to pandas Timestamp
        tmp['time_open'] = tmp['time_open'].apply(lambda ts: pd.Timestamp(ts, unit='ms'))

        tmp.set_index('time_open', inplace=True, drop=True)
        tmp.drop_duplicates(inplace=True)

        tmp['price_close_next_24H_max'] = tmp['price_close'].shift(periods=-24, freq="H").rolling(
            "24H", min_periods=24).max()
        tmp['ror_next_24H_max'] = tmp['price_close_next_24H_max'] / tmp['price_close']
        tmp['y_classification'] = tmp['ror_next_24H_max'].transform(lambda x: 1 if x >= 1.1 else 0)
        # tmp.drop(columns=['price_close_next_24H_max', 'ror_next_24H_max'])
        tmp.dropna(inplace=True)

        df_train = tmp.loc[tmp.index < cutoff_time]
        df_validation = tmp.loc[tmp.index >= cutoff_time]
        train_size.append(len(df_train))
        val_size.append(len(df_validation))

        if len(df_train) > 168:
            dict_train[identifier] = df_train[
                [
                    'price_open',
                    'price_high',
                    'price_low',
                    'price_close',
                    'volume_quote_currency',
                    'btc_price_open',
                    'btc_price_high',
                    'btc_price_low',
                    'btc_price_close',
                    'btc_volume_quote_currency',
                    'y_classification'
                ]
            ].to_numpy(dtype=np.float32)

        if len(df_validation) > 168:
            dict_validation[identifier] = df_validation[
                [
                    'price_open',
                    'price_high',
                    'price_low',
                    'price_close',
                    'volume_quote_currency',
                    'btc_price_open',
                    'btc_price_high',
                    'btc_price_low',
                    'btc_price_close',
                    'btc_volume_quote_currency',
                    'y_classification'
                ]
            ].to_numpy(dtype=np.float32)

    with open(os.path.join(os.path.dirname(__file__), TRAIN_DATA), 'wb') as f:
        pickle.dump(dict_train, f)

    with open(os.path.join(os.path.dirname(__file__), VALIDATION_DATA), 'wb') as f:
        pickle.dump(dict_validation, f)


if __name__ == "__main__":
    generate_pkl()
