"""
Eval test input with a trained model
"""
import os
import pickle
import statistics
import torch
import torch.nn.functional as F

from collections import defaultdict
from contextlib import nullcontext
from model import GPTConfig, GPT
from torchmetrics.classification import BinaryFBetaScore
# -----------------------------------------------------------------------------
init_file = 'ckpt.pt'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
seed = 421
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster

dataset = 'ccxt_kucoin_ohlcv_20230101_20230812'
val_filename = 'val.pkl'

beta = 0.025
threshold = 0.5
eps = 1e-7

exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, init_file)
checkpoint = torch.load(ckpt_path, map_location='cuda')  # {'cuda:0': 'cpu'}
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

block_size = gptconf.block_size
batch_size = checkpoint.get('config')['batch_size']

if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# get validation data
print("getting data...")
data_dir = os.path.join('data', dataset)
with open(os.path.join(data_dir, val_filename), 'rb') as f:
    data = pickle.load(f)
print("data loaded.")


def get_data():
    x_list = []
    y_list = []

    for symbol in data.keys():
        arr_symbol = data[symbol]
        for i in range(len(arr_symbol) - block_size + 1):
            x_single = torch.from_numpy(arr_symbol[i:i + block_size, :-1])
            x_single[:, [4, 9]] = torch.log(x_single[:, [4, 9]] + 1.0)
            x_min = torch.min(x_single, dim=0).values
            x_max = torch.max(x_single, dim=0).values
            x_single = (x_single - x_min) / (x_max - x_min + eps)
            x_list.append(x_single)

            y_single = torch.from_numpy(arr_symbol[i:i + block_size, -1])

            y_list.append(y_single)

            if len(x_list) >= batch_size:
                x = torch.stack(x_list)
                y = torch.stack(y_list)[:, [-1]]
                # print(f"Shape of x:{x.shape} | y:{y.shape}")
                if device_type == 'cuda':
                    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
                else:
                    x, y = x.to(device), y.to(device)
                yield x, y
                x_list = []
                y_list = []

    x = torch.stack(x_list)
    y = torch.stack(y_list)[:, [-1]]
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    yield x, y


def evaluate():
    # run evaluation
    with torch.no_grad():
        with ctx:
            f_betas = list()  # defaultdict(list)
            for i, (X, Y) in enumerate(get_data()):
                print(f'X:{X.shape}, Y:{Y.shape}')
                logits, _, _ = model(X)
                logits_flat = logits.view(-1, logits.size(-1))

                targets_flat = Y.view(-1, 1)

                # for thrs in range(1, 10):
                f_beta_func = BinaryFBetaScore(
                    beta=beta,
                    threshold=threshold
                ).to(device)
                fbeta = f_beta_func(logits_flat, targets_flat)
                f_betas.append(fbeta.item())

                if i % 1000 == 0:
                    print(f'{i}: f_beta = {statistics.mean(f_betas)}')

            # for thrs in range(1, 10):
            print(f'final f_beta = {statistics.mean(f_betas)}')


if __name__ == "__main__":
    evaluate()
