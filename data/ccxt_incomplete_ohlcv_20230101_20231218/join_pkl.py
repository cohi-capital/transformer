"""
Prepare KuCoin OHLCV data for binary classification, where targets are cases
with 10% price jump or higher.

We separate data in test and validation and put them into numpy arrays on a
per symbol basis.

Floats are 32-bit. Data is stored in pickle files.
"""
import argparse
import glob
import humanize
import joblib
import json
import os
import pickle
import sys

from datetime import datetime
from pathlib import Path


DAYS_IN_WEEK = 7
HOURS_IN_DAY = 24
N_SPLITS = 15
BLOCK_SIZE = 3 * HOURS_IN_DAY  # 24 * 3, 3 days
EPS = 1e-8
LOG_EPS = 1

WEEKLY_DECAY = 1 
TARGET_POS_SAMPLE_RATIO = 0.0  # 0.08 <= x < 1

MARKET_DATA_FILTERED_CCXT_KUCOIN = 'market_data_filtered_incomplete_ccxt_only_20230101_20231218.parquet'


def generate_pkl(data_type='both', block_size=BLOCK_SIZE, weekly_decay=WEEKLY_DECAY,
                 pos_sample_ratio=TARGET_POS_SAMPLE_RATIO, market_data_path=MARKET_DATA_FILTERED_CCXT_KUCOIN):
    filename_suffix = f'_block_{block_size}'
    if weekly_decay < 1:
        filename_suffix += f'_decay0{int(weekly_decay * 10)}'
    if pos_sample_ratio > 0:
        filename_suffix += f'_balance0{int(pos_sample_ratio * 100)}'

    train_data_path = f'train{filename_suffix}.joblib'
    val_data_path = f'val{filename_suffix}.joblib'
    metadata_path = f'meta_{data_type}{filename_suffix}.pkl'

    is_train = data_type == 'both' or data_type == 'train'
    is_val = data_type == 'both' or data_type == 'val'

    pos_samples = 0
    neg_samples = 0
    train_list = []
    val_list = []

    if is_train:
        train_paths = [Path(f) for f in glob.glob(f"tensors/train/*.pkl")]

        for i, p in enumerate(train_paths):
            with open(p, 'rb') as f:
                if i % 100 == 0:
                    print(f'train: {datetime.now()} | {i} of {len(train_paths)} | '
                          f'size: {humanize.naturalsize(sys.getsizeof(train_list))}')
                tensor_list = pickle.load(f)
                for tensor in tensor_list:
                    label = bool(tensor[-1][-1])
                    if label:
                        pos_samples += 1
                    else:
                        neg_samples += 1
                train_list.extend(tensor_list)

        joblib.dump(train_list, train_data_path)

        # with open(os.path.join(os.path.dirname(__file__), train_data_path), 'wb') as f:
        #     pickle.dump(train_list, f)

    if is_val:
        val_paths = [Path(f) for f in glob.glob(f"tensors/val/*.pkl")]

        for i, p in enumerate(val_paths):
            with open(p, 'rb') as f:
                if i % 100 == 0:
                    print(f'train: {datetime.now()} | {i} of {len(val_paths)} | '
                          f'size: {humanize.naturalsize(sys.getsizeof(val_list))}')
                    tensor_list = pickle.load(f)
                for tensor in tensor_list:
                    label = bool(tensor[-1][-1])
                    if label:
                        pos_samples += 1
                    else:
                        neg_samples += 1
                val_list.extend(tensor_list)

        joblib.dump(val_list, val_data_path)

        # with open(os.path.join(os.path.dirname(__file__), validation_data_path), 'wb') as f:
        #         pickle.dump(val_list, f)

    # Save the meta information as well to help us determine parameters later
    meta = {
        # 'train_size': train_size,
        # 'val_size': val_size,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='both')
    args = parser.parse_args()

    generate_pkl(args.dtype)
