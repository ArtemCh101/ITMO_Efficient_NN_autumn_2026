import csv
import os
import sys
import time
import numpy as np
import torch

sys.path.append("./hw1")
from models import SmallCNN

try:
  import pynvml

  pynvml.nvmlInit()
  nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
  HAS_NVML = True
except Exception:
  HAS_NVML = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def get_power_usage():
  if HAS_NVML:
    try:
      return pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
    except Exception:
      return None
  return None


def run_measurements():
  os.makedirs("hw1/results", exist_ok=True)
  csv_path = "hw1/results/measurements.csv"

  base_S = [32, 64, 128, 224, 256, 384, 512]
  base_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]

  np.random.seed(42)
  all_possible_S = [s for s in range(32, 513, 16) if s not in base_S]
  val_S = list(np.random.choice(all_possible_S, size=4, replace=False))

  all_possible_B = [b for b in range(1, 257) if b not in base_B]
  val_B = list(np.random.choice(all_possible_B, size=3, replace=False))

  grid_configs = []

  for s in base_S:
    for b in base_B:
      grid_configs.append((int(s), int(b), False))

  for s in val_S:
    for b in val_B:
      grid_configs.append((int(s), int(b), True))

  for s in base_S:
    for b in val_B:
      grid_configs.append((int(s), int(b), True))

  for s in val_S:
    for b in base_B:
      grid_configs.append((int(s), int(b), True))

  model = SmallCNN().cuda().eval()

  fieldnames = [
      "S",
      "B",
      "latency_s",
      "memory_bytes",
      "energy_j",
      "is_oom",
      "is_validation",
  ]

  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    total_cfg = len(grid_configs)
    print(
        f"Starting measurements on {torch.cuda.get_device_name(0)} for"
        f" {total_cfg} configurations...\n"
    )

    for idx, (s, b, is_val) in enumerate(grid_configs, 1):
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()

      val_str = "VAL" if is_val else "BASE"
      print(
          f"[{idx}/{total_cfg}] Testing S={s}, B={b} ({val_str})...",
          end="",
          flush=True,
      )

      try:
        x = torch.randn(b, 3, s, s, device="cuda", dtype=torch.float32)

        with torch.inference_mode():
          for _ in range(5):
            _ = model(x)
          torch.cuda.synchronize()

          warmup_power = get_power_usage()

          latencies = []
          powers = []

          for _ in range(20):
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append(t1 - t0)
            p = get_power_usage()
            if p is not None:
              powers.append(p)

        peak_mem = torch.cuda.max_memory_allocated()
        med_lat = float(np.median(latencies))

        if len(powers) > 0 and any(p is not None for p in powers):
          avg_p = float(np.mean([p for p in powers if p is not None]))
          energy_j = avg_p * med_lat
        else:
          energy_j = float("nan")

        writer.writerow({
            "S": s,
            "B": b,
            "latency_s": med_lat,
            "memory_bytes": peak_mem,
            "energy_j": energy_j,
            "is_oom": False,
            "is_validation": is_val,
        })
        f.flush()

        print(
            f" Done | Latency: {med_lat*1000:.2f} ms | Peak Mem:"
            f" {peak_mem/(1024**2):.1f} MB"
        )

      except torch.cuda.OutOfMemoryError:
        writer.writerow({
            "S": s,
            "B": b,
            "latency_s": float("nan"),
            "memory_bytes": float("nan"),
            "energy_j": float("nan"),
            "is_oom": True,
            "is_validation": is_val,
        })
        f.flush()
        print(" OOM Catch!")

      except Exception as e:
        print(f" Error: {e}")

  print(f"\nMeasurements complete! Saved to {csv_path}")


if __name__ == "__main__":
  run_measurements()
