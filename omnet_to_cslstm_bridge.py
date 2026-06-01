"""
omnet_to_cslstm_bridge.py
=========================
Converts OMNeT++/SUMO simulation data into the tensor format required
by the CS-LSTM (highwayNet) model, and converts predictions back to
metric coordinates for MPC.

KEY FIX in this version:
  The nbrs tensor must have AT LEAST 3 rows for the 3x1 conv kernel
  in the social pooling layer. When fewer real neighbours exist, we
  pad with dummy zero rows so the convolution always succeeds.
"""

import numpy as np
import torch
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────
M_TO_FT       = 3.28084
FT_TO_M       = 1.0 / M_TO_FT
LANE_WIDTH_M  = 3.5
IN_LEN        = 16
OUT_LEN       = 25
DT_MODEL_IN   = 0.1
DT_MODEL_OUT  = 0.2
GRID_LONG     = 13
GRID_LAT      = 3
CELL_LONG_FT  = 25.0
CELL_LAT_FT   = 12.0
ENC_SIZE      = 64



# ── Step 1: Build lane lateral-offset map ────────────────────────────────────
def build_lane_lateral_map(all_rows):
    """
    Maps every Lane_ID to a lateral offset in meters.
    Lanes with similar GlobalY centroids are parallel lanes on the same road.
    Each parallel lane step = LANE_WIDTH_M = 3.5 m.
    """
    lane_gy = defaultdict(list)
    for r in all_rows:
        lane_gy[float(r['Lane_ID'])].append(float(r['sigGlobalY']))

    lane_gy_mean = {
        lid: np.mean(gys)
        for lid, gys in lane_gy.items()
        if len(gys) >= 5
    }

    if not lane_gy_mean:
        return {float(lid): 0.0 for lid in lane_gy}, 1.0

    GY_SCALE = LANE_WIDTH_M / 3.20          # ~1.094 m per GlobalY unit
    GY_CLUSTER_THRESH = 20.0 / GY_SCALE     # cluster lanes within ~20 m

    # Group lane IDs whose GlobalY centroids are close together
    assigned = {}
    group_id = 0
    for lid in sorted(lane_gy_mean.keys()):
        placed = False
        for gid, members in assigned.items():
            ref_gy = np.mean([lane_gy_mean[m] for m in members])
            if abs(lane_gy_mean[lid] - ref_gy) < GY_CLUSTER_THRESH:
                assigned[gid].append(lid)
                placed = True
                break
        if not placed:
            assigned[group_id] = [lid]
            group_id += 1

    # Assign lateral offsets within each road group (sort by GlobalY)
    lane_map = {}
    for gid, members in assigned.items():
        members_sorted = sorted(members, key=lambda l: lane_gy_mean[l])
        for rank, lid in enumerate(members_sorted):
            lane_map[lid] = rank * LANE_WIDTH_M

    # Fallback for any lane not covered
    for lid in lane_gy.keys():
        if float(lid) not in lane_map:
            lane_map[float(lid)] = 0.0

    return lane_map, GY_SCALE


# ── Step 2: Temporal upsampling 1 Hz → 10 Hz ─────────────────────────────────
def upsample_1hz_to_10hz(times_s, local_y_m, vel_ms):
    """
    Linearly interpolates a 1 Hz trajectory to 10 Hz (0.1 s steps).
    sigLocalY is used as the primary signal — it is smooth cumulative
    road distance in real meters (verified: dLocalY == vel * dt).
    """
    times_s   = np.array(times_s,   dtype=float)
    local_y_m = np.array(local_y_m, dtype=float)
    vel_ms    = np.array(vel_ms,    dtype=float)

    t_10hz   = np.arange(times_s[0], times_s[-1] + 1e-9, DT_MODEL_IN)
    ly_10hz  = np.interp(t_10hz, times_s, local_y_m)
    vel_10hz = np.interp(t_10hz, times_s, vel_ms)

    return t_10hz, ly_10hz, vel_10hz


# ── Step 3: Build all model input tensors ────────────────────────────────────
def build_model_tensors(ego_id, all_rows, current_time_s, lane_map):
    car_rows = defaultdict(list)
    for r in all_rows:
        if float(r['Time_sec']) <= current_time_s + 1e-6:
            car_rows[r['Car_ID']].append(r)
    for cid in car_rows:
        car_rows[cid].sort(key=lambda r: float(r['Time_sec']))

    ego_data = car_rows.get(ego_id, [])
    if len(ego_data) < 2:
        raise ValueError(f"Ego car {ego_id} needs at least 2 seconds of history.")

    def _get_hist(rows):
        times = [float(r['Time_sec']) for r in rows]
        ly    = [float(r['sigLocalY']) for r in rows]
        vel   = [float(r['sigVel'])    for r in rows]
        if len(times) < 2: return None
        t10, ly10, _ = upsample_1hz_to_10hz(times, ly, vel)
        valid = t10 <= current_time_s + 1e-6
        ly10  = ly10[valid]
        if len(ly10) < IN_LEN:
            pad  = IN_LEN - len(ly10)
            ly10 = np.concatenate([np.full(pad, ly10[0]), ly10])
        return ly10[-IN_LEN:]

    # ── Ego history ──
    ego_ly_hist = _get_hist(ego_data)
    ego_ly_now  = ego_ly_hist[-1]
    ego_lane_id = float(ego_data[-1]['Lane_ID'])
    ego_lat_m   = lane_map.get(ego_lane_id, 0.0)
    ego_vel_now = float(ego_data[-1]['sigVel'])

    ego_long_ft = (ego_ly_hist - ego_ly_now) * M_TO_FT
    ego_lat_ft  = np.zeros(IN_LEN)
    ego_traj    = np.stack([ego_lat_ft, ego_long_ft], axis=1)
    hist = torch.FloatTensor(ego_traj).unsqueeze(1)

    # ── Neighbour Processing (Sorted & Filtered) ──
    valid_neighbors = []
    long_centers = [(i - GRID_LONG // 2) * CELL_LONG_FT for i in range(GRID_LONG)]
    lat_centers  = [(j - GRID_LAT  // 2) * CELL_LAT_FT  for j in range(GRID_LAT)]

    for cid, rows in car_rows.items():
        if cid == ego_id or len(rows) < 2:
            continue

        latest      = rows[-1]
        n_lane_id   = float(latest['Lane_ID'])
        n_lat_m     = lane_map.get(n_lane_id, 0.0)
        n_ly_now    = float(latest['sigLocalY'])

        lat_ft = (n_lat_m - ego_lat_m) * M_TO_FT
        lon_ft = (n_ly_now - ego_ly_now) * M_TO_FT

        r_best = min(range(GRID_LONG), key=lambda r: abs(lon_ft - long_centers[r]))
        c_best = min(range(GRID_LAT),  key=lambda c: abs(lat_ft - lat_centers[c]))

        # ONLY keep neighbors that actually fit inside the social grid
        if (abs(lon_ft - long_centers[r_best]) <= CELL_LONG_FT / 2 and
            abs(lat_ft - lat_centers[c_best]) <= CELL_LAT_FT / 2):

            n_ly_hist = _get_hist(rows)
            if n_ly_hist is not None:
                valid_neighbors.append({
                    'c_best': c_best,
                    'r_best': r_best,
                    'ly_hist': n_ly_hist,
                    'lat_ft': lat_ft
                })

    # CRITICAL FIX: Sort neighbors so they perfectly align with PyTorch's masked_scatter_ memory layout
    valid_neighbors.sort(key=lambda n: (n['c_best'], n['r_best']))

    # CRITICAL FIX: Swapped dimensions to [1, LAT, LONG, ENC_SIZE] to match model.py permute math
    masks = torch.zeros(1, GRID_LAT, GRID_LONG, ENC_SIZE)
    nbr_trajs_ft = []

    for n in valid_neighbors:
        n_long_ft = (n['ly_hist'] - ego_ly_now) * M_TO_FT
        n_lat_ft  = np.full(IN_LEN, n['lat_ft'])
        nbr_trajs_ft.append(np.stack([n_lat_ft, n_long_ft], axis=1))
        masks[0, n['c_best'], n['r_best'], :] = 1.0

    # Fallback to prevent PyTorch from crashing if the road is 100% empty
    if len(nbr_trajs_ft) == 0:
        nbr_trajs_ft.append(np.zeros((IN_LEN, 2)))
        # Mask remains completely zeroed out, so PyTorch math ignores it completely

    nbrs = torch.FloatTensor(np.stack(nbr_trajs_ft, axis=1))

    lat_enc = torch.zeros(1, 3)
    lon_enc = torch.zeros(1, 2)

    meta = {
        'ego_ly_now_m':   ego_ly_now,
        'ego_lat_now_m':  ego_lat_m,
        'ego_vel_now_ms': ego_vel_now,
        'ego_lane_id':    ego_lane_id,
        'current_time_s': current_time_s,
        'n_real_nbrs':    len(valid_neighbors),
    }

    return hist, nbrs, masks, lat_enc, lon_enc, meta

# ── Step 4: Select best maneuver ─────────────────────────────────────────────
def select_best_maneuver(lat_pred, lon_pred):
    """
    Picks the highest-probability maneuver from the 6 candidates.

    Index mapping (k=lon class, l=lat class, loop order k then l):
        0: maintain + right     3: brake + right
        1: maintain + straight  4: brake + straight
        2: maintain + left      5: brake + left
    """
    lat_p = torch.softmax(lat_pred, dim=1)[0].detach().cpu().numpy()
    lon_p = torch.softmax(lon_pred, dim=1)[0].detach().cpu().numpy()

    NAMES = [
        'Maintain_Right', 'Maintain_Straight', 'Maintain_Left',
        'Brake_Right',    'Brake_Straight',    'Brake_Left',
    ]
    probs    = {}
    idx      = 0
    best_idx = 0
    best_p   = -1.0

    for k in range(2):
        for l in range(3):
            p = float(lat_p[l] * lon_p[k])
            probs[NAMES[idx]] = p
            if p > best_p:
                best_p   = p
                best_idx = idx
            idx += 1

    return best_idx, probs


# ── Step 5: Convert predictions back to metric coordinates ───────────────────
def predictions_to_metric(fut_pred_list, best_maneuver_idx, meta):
    """
    Converts CS-LSTM output (local frame, feet, cumulative) to metric.

    Model output channels (after outputActivation):
        0: mux     lateral cumulative position (feet)
        1: muy     longitudinal cumulative position (feet)
        2: log_sx  log of lateral std dev (feet)
        3: log_sy  log of longitudinal std dev (feet)
        4: rho     correlation coefficient

    Returns dict with 25-step arrays in metric units.
    """
    traj_np = fut_pred_list[best_maneuver_idx][:, 0, :].detach().cpu().numpy()

    lat_ft      = traj_np[:, 0]
    long_ft     = traj_np[:, 1]
    sig_lat_ft  = np.exp(traj_np[:, 2])
    sig_long_ft = np.exp(traj_np[:, 3])

    ego_ly_now  = meta['ego_ly_now_m']
    ego_lat_now = meta['ego_lat_now_m']
    t_now       = meta['current_time_s']

    pred_long_m = ego_ly_now  + long_ft * FT_TO_M
    pred_lat_m  = ego_lat_now + lat_ft  * FT_TO_M
    sig_long_m  = np.clip(sig_long_ft * FT_TO_M, 0.0, 50.0)
    sig_lat_m   = np.clip(sig_lat_ft  * FT_TO_M, 0.0, 10.0)
    time_out    = t_now + np.arange(1, OUT_LEN + 1) * DT_MODEL_OUT

    return {
        'long_m':       pred_long_m,
        'lat_m':        pred_lat_m,
        'sigma_long_m': sig_long_m,
        'sigma_lat_m':  sig_lat_m,
        'time_s':       time_out,
    }


# ── Master function ───────────────────────────────────────────────────────────
def run_inference(net, ego_id, all_rows, current_time_s, lane_map):
    """
    End-to-end: OMNeT++ rows → tensors → CS-LSTM → metric predictions.

    Args:
        net            : loaded highwayNet (eval mode, train_flag=False)
        ego_id         : Car_ID string of the ego vehicle
        all_rows       : list of all CSV row dicts
        current_time_s : current simulation time in seconds
        lane_map       : from build_lane_lateral_map()

    Returns:
        pred_dict      : dict with long_m, lat_m, sigma_long_m, sigma_lat_m, time_s
        maneuver_probs : dict {name -> probability}
    """
    hist, nbrs, masks, lat_enc, lon_enc, meta = build_model_tensors(
        ego_id, all_rows, current_time_s, lane_map
    )

    with torch.no_grad():
        fut_pred, lat_pred, lon_pred = net(hist, nbrs, masks, lat_enc, lon_enc)

    best_idx, maneuver_probs = select_best_maneuver(lat_pred, lon_pred)
    pred_dict = predictions_to_metric(fut_pred, best_idx, meta)

    return pred_dict, maneuver_probs
