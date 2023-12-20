"""
Prepare KuCoin OHLCV data for binary classification, where targets are cases
with 10% price jump or higher.

We separate data in test and validation and put them into numpy arrays on a
per symbol basis.

Floats are 32-bit. Data is stored in pickle files.
"""
import argparse
import glob
import json
import os
import pickle
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

DAYS_IN_WEEK = 7
HOURS_IN_DAY = 24
N_SPLITS = 15
BLOCK_SIZE = 3 * HOURS_IN_DAY  # 24 * 3, 3 days
EPS = 1e-8
LOG_EPS = 1

WEEKLY_DECAY = 1 
TARGET_POS_SAMPLE_RATIO = 0.0  # 0.08 <= x < 1

MARKET_DATA_FILTERED_CCXT_INCOMPETE = 'market_data_filtered_incomplete_ccxt_only_20230101_20231218.parquet'

data_type = 'train'

filename_suffix = f'_block_{BLOCK_SIZE}'
if WEEKLY_DECAY < 1:
    filename_suffix += f'_decay0{int(WEEKLY_DECAY * 10)}'
if TARGET_POS_SAMPLE_RATIO > 0:
    filename_suffix += f'_balance0{int(TARGET_POS_SAMPLE_RATIO * 100)}'

train_data_path = f'data/train{filename_suffix}.pkl'
validation_data_path = f'data/val{filename_suffix}.pkl'
metadata_path = f'data/meta_{data_type}{filename_suffix}.pkl'

is_train = data_type == 'both' or data_type == 'train'
is_val = data_type == 'both' or data_type == 'val'

# Add bitcoin information to each row
print(f"{datetime.now()} | Loading data.")
df_base = pd.read_parquet(MARKET_DATA_FILTERED_CCXT_INCOMPETE)
print(f"{datetime.now()} | Data loaded.")

btc = df_base.loc[df_base['identifier'] == 'kucoin-BTC-USDT'].copy()
btc.columns = 'btc_' + btc.columns.values

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

df_merged['volume_quote_currency'] = df_merged['volume_quote_currency'].apply(lambda x: np.log(x + LOG_EPS))
df_merged['btc_volume_quote_currency'] = df_merged['btc_volume_quote_currency'].apply(lambda x: np.log(x + LOG_EPS))
print(f"{datetime.now()} | Data merge completed.")

# Get cut-off time between test and validation
min_date = pd.Timestamp(df_merged['time_open'].min(), unit='ms')
max_date = pd.Timestamp(df_merged['time_open'].max(), unit='ms')
time_between = max_date - min_date + pd.Timedelta(hours=1)
cutoff_time = min_date + (time_between * (N_SPLITS - 1) / N_SPLITS)


def generate_tensors(identifier):
    identifier_train_list = []
    identifier_val_list = []
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
    tmp.dropna(inplace=True)

    df_train = tmp.loc[tmp.index < cutoff_time].copy()
    df_validation = tmp.loc[tmp.index >= cutoff_time].copy()

    if is_train and (len(df_train) > BLOCK_SIZE):
        # Filter out negative samples to balance classes
        # classification_shifted = df_train['y_classification'].shift(-BLOCK_SIZE)
        # classifications_counts = classification_shifted.value_counts()
        # pos_counts = classifications_counts[1] if 1 in classifications_counts else 0
        # neg_counts = classifications_counts[0] if 0 in classifications_counts else 0
        # # Did some arithmetic to come up with this
        # neg_scalar = (pos_counts - pos_counts * TARGET_POS_SAMPLE_RATIO) / (neg_counts * TARGET_POS_SAMPLE_RATIO)
        # df_train['rand1'] = np.random.random(len(df_train))
        # # Keep negative values if this condition is met
        # df_train['include'] = (~classification_shifted.isna()) & (
        #         (classification_shifted == 1) | (df_train['rand1'] < neg_scalar)
        # )
        # Override inclusion decisions up to this point
        df_train['include'] = True

        # Filter out older data
        df_train['decay'] = WEEKLY_DECAY ** (
                    (cutoff_time - df_train.index).days // DAYS_IN_WEEK)
        df_train['rand2'] = np.random.random(len(df_train))
        df_train['include'] = df_train['include'] & (df_train['decay'] >= df_train['rand2'])

        # Do not consider last BLOCK_SIZE indices for training so we can have full training blocks.
        # df_train['include'] = True
        mask_batch_size = [True if j > len(df_train) - BLOCK_SIZE else False for j in range(len(df_train))]
        df_train.loc[mask_batch_size, 'include'] = False

        df_train.reset_index(inplace=True)
        df_train['key'] = df_train['time_open'].transform(lambda x: f"{identifier} | {x}")
        df_train.set_index('key', inplace=True)

        # Get indices for rows to extra
        starting_indices = [df_train.index.get_loc(idx) for idx, _ in df_train.loc[df_train.include].iterrows()]

        # Drop unnecessary columns
        df_train = df_train[
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
        ]

        for starting_idx in starting_indices:
            block = df_train.iloc[starting_idx:starting_idx + BLOCK_SIZE]
            label = block['y_classification'][-1]
            # if label:
            #     pos_samples += 1
            # else:
            #     neg_samples += 1

            train_tensor = torch.tensor(block.values, dtype=torch.float32)
            x_min = torch.min(train_tensor, dim=0).values
            x_max = torch.max(train_tensor, dim=0).values
            train_tensor = (train_tensor - x_min) / (x_max - x_min + EPS)
            
            identifier_train_list.append(train_tensor)
        
        with open(f'tensors/train/{identifier}.pkl', 'wb') as f:
            pickle.dump(identifier_train_list, f)

            # pos_samples += train_tensor[:, -1].count_nonzero().values
            # neg_samples += BLOCK_SIZE - train_tensor[:, -1].count_nonzero().values

    if is_val and (len(df_validation) > BLOCK_SIZE):
        df_validation = df_validation[
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
        ]

        for starting_idx in range(len(df_validation) - BLOCK_SIZE):
            block = df_validation.iloc[starting_idx:starting_idx + BLOCK_SIZE]
            label = block['y_classification'][-1]
            # if label:
            #     pos_samples += 1
            # else:
            #     neg_samples += 1

            val_tensor = torch.tensor(block.values, dtype=torch.float32)
            x_min = torch.min(val_tensor, dim=0).values
            x_max = torch.max(val_tensor, dim=0).values
            val_tensor = (val_tensor - x_min) / (x_max - x_min + EPS)
            
            identifier_val_list.append(val_tensor)
            
        with open(f'tensors/val/{identifier}_{i}.pkl', 'wb') as f:
            pickle.dump(identifier_val_list, f)
            
            # pos_samples += val_tensor[:, -1].count_nonzero().values
            # neg_samples += BLOCK_SIZE - val_tensor[:, -1].count_nonzero().values


def generate_pkl():
    # Go through symbol by symbol to generate training and validation data
    train_list = []
    val_list = []

    total_symbols = len(df_merged.identifier.unique())
    start_time = datetime.now()

    pos_samples = 0
    neg_samples = 0

    # no_pos_sample = 0
    # one_plus_pos_sample = 0
    # last_pos_sample = 0
    
    trading_pairs = df_merged.identifier.unique()

    # Create a multiprocessing Pool
    pool = Pool(2)                         
    pool.map(generate_tensors, trading_pairs)
    
    if is_train:
        train_paths = [Path(f) for f in glob.glob(f"tensors/train/*.pkl")]

        for p in paths:
            with open(p, 'rb') as f:
                tensor_list = pickle.load(f)
                for tensor in tensor_list:
                    label = bool(tensor[-1][-1])
                    if label:
                        pos_samples += 1
                    else:
                        neg_samples += 1
                train_list.extend(tensor_list)

        with open(os.path.join(os.path.dirname(__file__), train_data_path), 'wb') as f:
            pickle.dump(train_list, f)

    if is_val:
        val_paths = [Path(f) for f in glob.glob(f"tensors/val/*.pkl")]

        for p in paths:
            with open(p, 'rb') as f:
                tensor_list = pickle.load(f)
                for tensor in tensor_list:
                    label = bool(tensor[-1][-1])
                    if label:
                        pos_samples += 1
                    else:
                        neg_samples += 1
                val_list.extend(tensor_list)

        with open(os.path.join(os.path.dirname(__file__), validation_data_path), 'wb') as f:
                pickle.dump(val_list, f)


    # Save the meta information as well to help us determine parameters later
    meta = {
        'train_size': len(train_list),
        'val_size': len(val_list),
        'pos_size': pos_samples,
        'neg_size': neg_samples,
        'pos_ratio': pos_samples / (pos_samples + neg_samples),
        # 'no_pos_sample': no_pos_sample,
        # 'one_plus_pos_sample': one_plus_pos_sample,
        # 'last_pos_sample': last_pos_sample
    }
    print(meta)

    with open(os.path.join(os.path.dirname(__file__), f'{metadata_path}'), 'w') as f:
        json.dump(meta, f)


if __name__ == "__main__":
    generate_pkl()
