import json
import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

sys.path.append("./hw1")
from equations import memory, latency, energy, flops, bytes_moved

def fit_calibration():
    csv_path = "hw1/results/measurements.csv"
    json_path = "hw1/results/theta.json"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run measure.py first.")
        return

    df = pd.read_csv(csv_path)
    valid_df = df[df["is_oom"] == False].copy()

    train_df = valid_df[valid_df["is_validation"] == False].copy()
    val_df = valid_df[valid_df["is_validation"] == True].copy()

    print(f"Loaded {len(valid_df)} valid measurements ({len(train_df)} train, {len(val_df)} val).\n")

    def calc_mape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        mask = y_true > 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    # 1. Memory Model Evaluation (Analytic)
    print("--- 1. Evaluating Memory Model ---")
    mem_pred_train = memory(train_df["S"].values, train_df["B"].values)
    mem_pred_val = memory(val_df["S"].values, val_df["B"].values)

    mem_mape_train = calc_mape(train_df["memory_bytes"].values, mem_pred_train)
    mem_mape_val = calc_mape(val_df["memory_bytes"].values, mem_pred_val)

    # 2. Fitting Latency Model (theta_0, theta_bw, theta_gflops)
    print("\n--- 2. Fitting Latency Model ---")
    def lat_wrapper(SB, theta_0, theta_bw, theta_gflops):
        S, B = SB
        return latency(S, B, (theta_0, theta_bw, theta_gflops))

    p0_lat = [1e-3, 1e11, 1e12]
    bounds_lat = (0, [np.inf, np.inf, np.inf])

    popt_lat, _ = curve_fit(
        lat_wrapper,
        (train_df["S"].values, train_df["B"].values),
        train_df["latency_s"].values,
        p0=p0_lat,
        bounds=bounds_lat
    )

    theta_0, theta_bw, theta_gflops = popt_lat
    print(f"  Calibrated Latency Parameters:")
    print(f"    - Base Overhead (theta_0): {theta_0 * 1000:.3f} ms")
    print(f"    - Memory Bandwidth (theta_bw): {theta_bw / 1e9:.2f} GB/s")
    print(f"    - Compute Throughput (theta_gflops): {theta_gflops / 1e12:.2f} TFLOPS")

    lat_pred_train = lat_wrapper((train_df["S"].values, train_df["B"].values), *popt_lat)
    lat_pred_val = lat_wrapper((val_df["S"].values, val_df["B"].values), *popt_lat)

    lat_mape_train = calc_mape(train_df["latency_s"].values, lat_pred_train)
    lat_mape_val = calc_mape(val_df["latency_s"].values, lat_pred_val)

    # 3. Fitting Energy Model (P_idle, e_byte, e_flop)
    print("\n--- 3. Fitting Energy Model ---")
    energy_train = train_df.dropna(subset=["energy_j"])
    energy_train = energy_train[energy_train["energy_j"] > 0]

    if len(energy_train) > 0:
        def energy_wrapper(SB, P_idle, e_byte, e_flop):
            S, B = SB
            theta_energy = (theta_0, theta_bw, theta_gflops, P_idle, e_byte, e_flop)
            return energy(S, B, theta_energy)

        p0_energy = [10.0, 1e-9, 1e-11]
        bounds_energy = (0, [np.inf, np.inf, np.inf])

        popt_energy, _ = curve_fit(
            energy_wrapper,
            (energy_train["S"].values, energy_train["B"].values),
            energy_train["energy_j"].values,
            p0=p0_energy,
            bounds=bounds_energy
        )

        P_idle, e_byte, e_flop = popt_energy
        print(f"  Calibrated Energy Parameters:")
        print(f"    - Idle Power (P_idle): {P_idle:.2f} W")
        print(f"    - Energy per Byte (e_byte): {e_byte * 1e9:.3f} nJ/Byte")
        print(f"    - Energy per FLOP (e_flop): {e_flop * 1e12:.3f} pJ/FLOP")

        energy_val = val_df.dropna(subset=["energy_j"])
        energy_val = energy_val[energy_val["energy_j"] > 0]

        en_pred_train = energy_wrapper((energy_train["S"].values, energy_train["B"].values), *popt_energy)
        en_pred_val = energy_wrapper((energy_val["S"].values, energy_val["B"].values), *popt_energy)

        en_mape_train = calc_mape(energy_train["energy_j"].values, en_pred_train)
        en_mape_val = calc_mape(energy_val["energy_j"].values, en_pred_val)
    else:
        P_idle, e_byte, e_flop = 0.0, 0.0, 0.0
        en_mape_train, en_mape_val = float("nan"), float("nan")
        print("  No valid energy data available for calibration.")

    # Save Results
    results = {
        "theta": {
            "latency": {
                "theta_0": float(theta_0),
                "theta_bw": float(theta_bw),
                "theta_gflops": float(theta_gflops)
            },
            "energy": {
                "P_idle": float(P_idle),
                "e_byte": float(e_byte),
                "e_flop": float(e_flop)
            }
        },
        "metrics": {
            "memory_mape_train_pct": mem_mape_train,
            "memory_mape_val_pct": mem_mape_val,
            "latency_mape_train_pct": lat_mape_train,
            "latency_mape_val_pct": lat_mape_val,
            "energy_mape_train_pct": en_mape_train,
            "energy_mape_val_pct": en_mape_val
        }
    }

    os.makedirs("hw1/results", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print Summary Analysis
    print("\n================ CALIBRATION SUMMARY ================")
    print(f" Memory MAPE  | Train: {mem_mape_train:6.2f}% | Val: {mem_mape_val:6.2f}%")
    print(f" Latency MAPE | Train: {lat_mape_train:6.2f}% | Val: {lat_mape_val:6.2f}%")
    if not np.isnan(en_mape_train):
        print(f" Energy MAPE  | Train: {en_mape_train:6.2f}% | Val: {en_mape_val:6.2f}%")
    print("=====================================================")
    print(f"Results successfully saved to {json_path}")

if __name__ == "__main__":
    fit_calibration()
