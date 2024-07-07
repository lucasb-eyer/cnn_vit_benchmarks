# Run this through `untilfail.bash python main.py`, so it restarts a new process for each setting.

MODELS = {
    'ConvNeXt-B': 'convnext_base',
    'NFNet-F0': 'dm_nfnet_f0',
    'NFNet-F1': 'dm_nfnet_f1',
    'ViT-B/16': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
    # 'ViTDet-B/16w14': 'samvit_base_patch16.sa1b',  # Needs a timm patch, see patchfile.
}
IMAGE_SIZES = [128, 224, 256, 384, 448, 512, 768, 896, 1024]
BATCH_SIZES = [1, 8, 32]

# Equivalences:
# - NFNet-F1 is about ViT-B/16 (comparing "ScalingViT" and "ConvNets match ViTs" papers)
# - NFNet-F3 would be ViT-L/16
# - ConvNeXt-B: they chose to call it "B".



# import os
# os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

import torch
import time
import timm

from torchprofile import profile_macs
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No handlers found")
warnings.filterwarnings("ignore", category=UserWarning, message="TensorFloat32 tensor cores for float32 matrix.*")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @title Benchmarking utils

# With compile and all tricks in the compile tutorial
# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups
import torch._dynamo
def calculate_time2(model, x, runs=10):
  model.eval()
  torch._dynamo.reset()
  model_opt = torch.compile(model, mode="reduce-overhead")

  times = []
  for _ in range(runs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
      start.record()
      _ = model_opt(x)
      end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))
  del model_opt
  return min(times), times

def calculate_flops(model, x):
  model.eval()
  with torch.no_grad():
    macs = profile_macs(model, x)
  flops = 2 * macs  # 1 MAC operation = 2 FLOPs (1 multiplication + 1 addition)
  return flops

def calculate_params(model):
  return sum(p.numel() for p in model.parameters())

def memory_start():
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats(device=None)
  return torch.cuda.max_memory_allocated(device=None)

def memory_tick():
  return torch.cuda.max_memory_allocated(device=None)

def make_model(name, res):
  if 'samvit' in MODELS[name]:
      return timm.create_model(MODELS[name], img_size=res, window_size={128: 8, 224: 14, 256: 16, 384: 12, 448: 14, 512: 16, 768: 16, 896: 14, 1024: 16}[res])
  elif 'vit_base_patch16' in MODELS[name]:
    return timm.create_model(MODELS[name], img_size=res)
  else:
    return timm.create_model(MODELS[name])


def benchmark(model_name, res, batch_size=1, show=True, dtype=torch.float32, mode_for_log=None):
  print(f"\rStarting {model_name=} {res=} {batch_size=} {dtype=} {mode_for_log}...", end="", flush=True)
  try:
    x = torch.randn(batch_size, 3, res, res, dtype=dtype).to(device)

    m0 = memory_start()
    model = make_model(model_name, res).to(dtype).to(device).eval()
    m1 = memory_tick()
    time, times = calculate_time2(model, x, runs=20)
    m2 = memory_tick()
    mem_fwd = m2 - m1
    mem_tot = m2 - m0
    flops = calculate_flops(model, x)
    params = calculate_params(model)  # Note: params == mem_tot - mem_fwd * 4 for float32.
    return dict(time=time, flops=flops, params=params, mem_tot=mem_tot, mem_fwd=mem_fwd)
  except torch.cuda.OutOfMemoryError:
    print(f"\nOOM on {model_name=} {res=} {batch_size=} {dtype=}", flush=True)
    return dict(time=torch.nan, flops=torch.nan, params=torch.nan, mem_tot=torch.nan, mem_fwd=torch.nan)
  except Exception as e:  # dynamo raises a different error on OOM...
    if "OutOfMemoryError" in str(e):
      print(f"\nOOM on {model_name=} {res=} {batch_size=} {dtype=}", flush=True)
      return dict(time=torch.nan, flops=torch.nan, params=torch.nan, mem_tot=torch.nan, mem_fwd=torch.nan)
    raise


# Save/Load "database"
import json

def read_all(fname):
  try:
    with open(fname, 'r') as f:
      return json.load(f)
  except FileNotFoundError:
    return []

def isdone(fname, key, verbose=True):
  for row in read_all(fname):
    if all(str(row.get(k)) == str(v) for k, v in key.items()):
      if verbose:
        print(f'\rSkipping {key}', end='', flush=True)
      return True
  return False

def add_or_update(fname, key, values, verbose=True):
  rows = read_all(fname)
  for row in rows:
    if all(str(row.get(k)) == str(v) for k, v in key.items()):
      break
  else:
    rows.append({**key, **values})
  with open(fname, 'w+') as f:
    json.dump(rows, f)
  if verbose:
    print(f'\rDid {len(rows)} experiments (last done: {key})', flush=True)
  import sys ; sys.exit(0)


# The benchmarking loop
from torch.nn.attention import SDPBackend, sdpa_kernel

for m in MODELS:
  for res in IMAGE_SIZES:
    for bs in BATCH_SIZES:
      kernels = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if 'ViT' in m else [SDPBackend.MATH]
      for kernel in kernels:
        key = dict(gpu=torch.cuda.get_device_name(), m=m, res=res, bs=bs, kernel=str(kernel).replace('SDPBackend.', ''))
        with sdpa_kernel(kernel):
          if kernel != SDPBackend.FLASH_ATTENTION:  # Only can do (b)f16
            torch.set_float32_matmul_precision('highest')
            if not isdone('results.json', (k := {**key, 'dtype': 'fp32_max'})):
              add_or_update('results.json', key=k, values=benchmark(m, res=res, batch_size=bs, mode_for_log=kernel))
            torch.set_float32_matmul_precision('high')
            if not isdone('results.json', (k := {**key, 'dtype': 'fp32_hi'})):
              add_or_update('results.json', key=k, values=benchmark(m, res=res, batch_size=bs, mode_for_log=kernel))
          if not isdone('results.json', (k := {**key, 'dtype': 'bf16'})):
            add_or_update('results.json', key=k, values=benchmark(m, res=res, batch_size=bs, dtype=torch.bfloat16, mode_for_log=kernel))
          if not isdone('results.json', (k := {**key, 'dtype': 'f16'})):
            add_or_update('results.json', key=k, values=benchmark(m, res=res, batch_size=bs, dtype=torch.float16, mode_for_log=kernel))

import sys ; sys.exit(1)
