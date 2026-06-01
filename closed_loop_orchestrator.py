"""
closed_loop_orchestrator.py
===========================
The master execution engine that coordinates all 3 architectural layers:
  Layer 1: CS-LSTM Trajectory & Uncertainty Predictions (from predictions_metric.csv)
  Layer 2: Point-Mass Non-linear Model Predictive Control (nmpc_layer.py)
  Layer 3: Linear Half-Space Control Barrier Function Safety Filter (cbf_layer.py)
"""

import pandas as pd
import numpy as np
import ast
import time
from nmpc_layer import Layer2_NMPC
from cbf_layer import Layer3_CBF

class VehicleControlUnit:
    def __init__(self):
        # Instantiate Layer 2 and Layer 3 modules
        self.nmpc = Layer2_NMPC(N=25, dt=0.2)
        self.cbf  = Layer3_CBF(alpha1=1.5, alpha2=1.5)

    def run_step(self, current_state, final_goal, ego_predictions, neighbors_current, neighbors_predictions):
        """
        Executes a single synchronous execution tick across the pipeline layers.
        """
        # Step 1: Run Layer 2 (NMPC) to get the optimized nominal control plan
        nmpc_res = self.nmpc.solve(
            current_state=current_state,
            final_goal=final_goal,
            ego_data=ego_predictions,
            neighbors_data=neighbors_predictions
        )

        # Extract the first control step from the NMPC prediction horizon
        u_mpc = [nmpc_res['a_x'][0], nmpc_res['a_y'][0]]

        # Step 2: Run Layer 3 (CBF) to filter the control commands against immediate threats
        cbf_res = self.cbf.filter_control(
            current_state=current_state,
            u_mpc=u_mpc,
            neighbors_current=neighbors_current
        )

        return {
            "u_mpc": u_mpc,
            "u_safe": cbf_res['u_safe'],
            "mode": cbf_res['mode'],
            "nmpc_success": nmpc_res['success']
        }

def simulate_pipeline():
    print("="*70)
    print("RUNNING COMPLETE CLOSED-LOOP ARCHITECTURE SIMULATION")
    print("="*70)

    # 1. Load real prediction metrics from Layer 1
    df = pd.read_csv('predictions_metric.csv')
    vcu = VehicleControlUnit()

    # Let's track the vehicle step-by-step for the first 5 timestamps in the CSV
    unique_times = sorted(df['Time_s'].unique())[:5]

    for t in unique_times:
        print(f"\n[SIMULATION TIME: t = {t}s]")

        # Extract Ego Row
        ego_row = df[(df['Time_s'] == t) & (df['Car_ID'] == 1.0)].iloc[0]

        # Parse Ego Position and Speed States
        ego_pred_x = ast.literal_eval(ego_row['Pred_Lateral_m'])
        ego_pred_y = ast.literal_eval(ego_row['Pred_LocalY_m'])

        current_state = [
            ego_pred_x[0],                # current lateral position
            ego_row['Current_LocalY_m'],  # current longitudinal position
            0.0,                          # initial lateral velocity assumption
            ego_row['Current_Vel_ms']     # current forward velocity
        ]

        # Target destination point (100 meters ahead down the corridor)
        final_goal = [current_state[0], current_state[1] + 100.0]

        ego_predictions = {
            'Pred_Lateral_m': ego_pred_x,
            'Pred_LocalY_m': ego_pred_y
        }

        # 2. Dynamic Scenario Injector (Normal vs Critical Test)
        neighbors_current = []
        neighbors_predictions = []

        if t == unique_times[2]:
            # Inject a sudden critical hazard at the 3rd time step to test runtime switching
            print("  ! CRITICAL SCENARIO INJECTED: Slower vehicle cut-in detected !")

            # Immediate positions for Layer 3 CBF
            neighbors_current.append([current_state[0], current_state[1] + 4.0, 0.0, 1.0])

            # Future horizons for Layer 2 NMPC
            neighbors_predictions.append({
                'Pred_Lateral_m': [current_state[0]] * 25,
                'Pred_LocalY_m':  [current_state[1] + 4.0 + (1.0 * 0.2 * k) for k in range(25)],
                'Sigma_Lat_m':    [0.1] * 25,
                'Sigma_Long_m':   [0.2] * 25
            })
        else:
            # Normal driving conditions: obstacle is far away and safe
            print("  Standard driving scenario. Roadway clear.")
            neighbors_current.append([current_state[0] + 3.5, current_state[1] + 40.0, 0.0, current_state[3]])
            neighbors_predictions.append({
                'Pred_Lateral_m': [current_state[0] + 3.5] * 25,
                'Pred_LocalY_m':  [current_state[1] + 40.0 + (current_state[3] * 0.2 * k) for k in range(25)],
                'Sigma_Lat_m':    [0.1] * 25,
                'Sigma_Long_m':   [0.2] * 25
            })

        # 3. Fire commands through the integrated VCU architecture
        start_time = time.time()
        verdict = vcu.run_step(current_state, final_goal, ego_predictions, neighbors_current, neighbors_predictions)
        elapsed_ms = (time.time() - start_time) * 1000

        # 4. Output Step Verdicts
        print(f"  Execution Time  : {elapsed_ms:.2f} ms")
        print(f"  NMPC Target Acc : a_x={verdict['u_mpc'][0]:.3f} m/s^2, a_y={verdict['u_mpc'][1]:.3f} m/s^2")
        print(f"  CBF Final Output: a_x={verdict['u_safe'][0]:.3f} m/s^2, a_y={verdict['u_safe'][1]:.3f} m/s^2")
        print(f"  Safety Filter Mode: {verdict['mode']}")
        print("-" * 50)

if __name__ == "__main__":
    simulate_pipeline()
