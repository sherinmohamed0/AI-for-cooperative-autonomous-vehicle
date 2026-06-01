"""
evaluate_omnet.py
=================
Runs CS-LSTM inference on OMNeT++ simulation data.
Produces predictions_metric.csv with 25-step metric predictions for MPC.

Usage:
    python evaluate_omnet.py
"""

import csv
import torch
import pandas as pd
import numpy as np
from collections import Counter
from model import highwayNet
from omnet_to_cslstm_bridge import build_lane_lateral_map, run_inference

# ── Model configuration (must match trained weights) ──────────────────────────
args = {
    'use_cuda':             False,   # set True if you have a CUDA GPU
    'encoder_size':         64,
    'decoder_size':         128,
    'in_length':            16,
    'out_length':           25,
    'grid_size':            (13, 3),
    'soc_conv_depth':       64,
    'conv_3x1_depth':       16,
    'dyn_embedding_size':   32,
    'input_embedding_size': 32,
    'num_lat_classes':      3,
    'num_lon_classes':      2,
    'use_maneuvers':        True,
    'train_flag':           False,   # MUST be False for inference
}

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
net = highwayNet(args)
net.load_state_dict(torch.load('trained_models/cslstm_m.tar', map_location='cpu'))
net.eval()
print("Model loaded.")

# ── Load OMNeT++ data ─────────────────────────────────────────────────────────
print("Loading CSV data...")
all_rows = []
with open('Simulation_data.csv') as f:
    for row in csv.DictReader(f):
        all_rows.append(row)
print(f"Loaded {len(all_rows)} rows.")

# ── Build lane lateral map (one-time) ────────────────────────────────────────
lane_map, _ = build_lane_lateral_map(all_rows)
print(f"Lane map built: {len(lane_map)} lane IDs → lateral offsets (m)")

# ── Choose ego vehicle ────────────────────────────────────────────────────────
# Use the car with the most data points so we get the most predictions.
# You can change this to any specific Car_ID string, e.g. ego_id = '17.0'
ego_id = Counter(r['Car_ID'] for r in all_rows).most_common(1)[0][0]
print(f"Ego vehicle: Car {ego_id}")

# ── Get valid timestamps ──────────────────────────────────────────────────────
# Only process times where the ego car actually exists AND
# we have at least 2 seconds of history (needed for upsampling)
ego_times = sorted(
    float(r['Time_sec']) for r in all_rows if r['Car_ID'] == ego_id
)
# Need at least 2 real data points before we try to run inference
valid_times = [t for t in ego_times if t >= ego_times[0] + 2.0]
print(f"Processing {len(valid_times)} timestamps "
      f"(t={valid_times[0]:.0f}s to t={valid_times[-1]:.0f}s)")

# ── Main inference loop ───────────────────────────────────────────────────────
results = []
skipped = 0

for t in valid_times:
    try:
        pred, maneuver_probs = run_inference(net, ego_id, all_rows, t, lane_map)

        # Get ego state at this timestep for logging
        ego_now = next(
            (r for r in all_rows if r['Car_ID'] == ego_id
             and float(r['Time_sec']) == t), None
        )
        ego_lane = float(ego_now['Lane_ID']) if ego_now else -1
        ego_vel  = float(ego_now['sigVel'])  if ego_now else 0.0
        ego_ly   = float(ego_now['sigLocalY']) if ego_now else 0.0

        results.append({
            'Time_s':             t,
            'Car_ID':             ego_id,
            'Lane_ID':            ego_lane,
            'Current_LocalY_m':   ego_ly,
            'Current_Vel_ms':     ego_vel,
            **maneuver_probs,
            'Pred_LocalY_m':      pred['long_m'].tolist(),
            'Pred_Lateral_m':     pred['lat_m'].tolist(),
            'Sigma_Long_m':       pred['sigma_long_m'].tolist(),
            'Sigma_Lat_m':        pred['sigma_lat_m'].tolist(),
            'Pred_Times_s':       pred['time_s'].tolist(),
        })

    except Exception as e:
        skipped += 1
        if skipped <= 5:   # only print first 5 skip reasons to avoid clutter
            print(f"  t={t:.1f}s skipped: {e}")

    if len(results) % 50 == 0 and len(results) > 0:
        print(f"  Processed {len(results)} timesteps...")

# ── Save results ──────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv('predictions_metric.csv', index=False)

print(f"\nDone.")
print(f"  Saved    : {len(results)} predictions → predictions_metric.csv")
print(f"  Skipped  : {skipped} timesteps")

if results:
    r0  = results[0]
    ly  = r0['Pred_LocalY_m']
    vel = r0['Current_Vel_ms']
    best_m = max(
        (k for k in r0 if k in [
            'Maintain_Right','Maintain_Straight','Maintain_Left',
            'Brake_Right','Brake_Straight','Brake_Left'
        ]),
        key=lambda k: r0[k]
    )
    print(f"\nSample — first prediction at t={r0['Time_s']:.0f}s:")
    print(f"  Current vel   : {vel:.2f} m/s")
    print(f"  Best maneuver : {best_m} ({r0[best_m]:.3f})")
    print(f"  Pred LocalY   : now={r0['Current_LocalY_m']:.1f}m  "
          f"+1s={ly[4]:.1f}m  +3s={ly[14]:.1f}m  +5s={ly[24]:.1f}m")
    expected_1s = r0['Current_LocalY_m'] + vel * 1.0
    print(f"  Expected +1s  : ~{expected_1s:.1f}m  (vel × 1s sanity check)")
