import socket
import json
import time
import concurrent.futures
import torch

# ==============================================================================
# 1. FIXED IMPORTS (Matching your exact files)
# ==============================================================================
from model import highwayNet
from closed_loop_orchestrator import VehicleControlUnit
from omnet_to_cslstm_bridge import run_inference


# ==============================================================================
# 2. DEFINE NETWORK CONFIGURATION (Matching your train.py parameters)
# ==============================================================================
def get_model_arguments():
    args = {}
    args['use_cuda'] = False  # Set to False to run inference on your Ryzen CPU
    args['encoder_size'] = 64
    args['decoder_size'] = 128
    args['in_length'] = 16
    args['out_length'] = 25
    args['grid_size'] = (13, 3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 2
    args['use_maneuvers'] = True
    args['train_flag'] = False  # CRITICAL: Must be False for real-time evaluation
    return args


# ==============================================================================
# 3. TAILORED MODEL LOADER
# ==============================================================================
def load_pytorch_model(model_path):
    print("=" * 60)
    print(f"[*] INITIALIZING RECONSTRUCTION: {model_path}")
    print("=" * 60)
    try:
        # Get the strict configuration dictionary your network demands
        args = get_model_arguments()

        # Instantiate your blueprint architecture with its configurations
        net = highwayNet(args)

        # Load the state_dict weights directly into memory
        # map_location='cpu' forces it onto the Ryzen CPU cores for execution
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Pouing the weights into your architecture mapping
        net.load_state_dict(state_dict)

        # Lock model in Evaluation Mode (disables tracking behavior shifts)
        net.eval()
        print("[SUCCESS] highwayNet Architecture is armed and locked in memory!\n")
        return net

    except Exception as e:
        print(f"\n[FATAL ERROR INITIALIZING AI MODEL]: {e}")
        print("[Diagnostic] Verify that model.py is in the same directory as this script.")
        exit(1)


# ==============================================================================
# 4. CONCURRENT WORKER CORE (Dispatched across Ryzen CPU Threads)
# ==============================================================================
def compute_agent(ego_id, current_state, final_goal, ego_preds, nbrs_current, nbrs_preds, vcu_instance):
    """
    Executes Layer 2 (NMPC Optimizer) and Layer 3 (Control Barrier Functions)
    concurrently on individual CPU threads per vehicle.
    """
    try:
        verdict = vcu_instance.run_step(current_state, final_goal, ego_preds, nbrs_current, nbrs_preds)
        return ego_id, verdict, True
    except Exception as e:
        print(f"[Core Math Error - Vehicle {ego_id}]: {e}")
        # Fallback safe behavior: Hard deceleration, zero steering
        return ego_id, {"u_safe": [0.0, -6.0], "mode": "Emergency Fallback"}, False


# ==============================================================================
# 5. REAL-TIME CO-SIMULATION HIL SERVER
# ==============================================================================
def run_realtime_server():
    # Warm-up and pull PyTorch model into memory once using the correct filename
    model_file_path = 'trained_models/cslstm_m.tar'
    net = load_pytorch_model(model_file_path)

    # Build the Cross-Platform Network Boundary Socket (Tailscale Mesh Compatible)
    HOST = '0.0.0.0'  # Listens to all physical & Tailscale Virtual Network Interfaces
    PORT = 9999       # Strict pairing mapping to OMNeT++ target port

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((HOST, PORT))
    print(f"[Network] UDP HIL Daemon Listening globally on {HOST}:{PORT}")
    print("[-] Standing by. Launch the OMNeT++/Veins simulation on the Linux VM...")
    print("=" * 60)

    # State tracking metrics
    active_agents = {}
    live_history_buffer = []

    # Map SUMO nominal lane strings to numerical lateral coordinate tracks (meters)
    lane_map = {"0": 0.0, "1": 3.5, "2": 7.0}

    # Provision worker pool scaling across up to 8 parallel math threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        while True:
            # ------------------------------------------------------------------
            # PHASE 1: SENSE (Awaiting incoming payload from Linux VM)
            # ------------------------------------------------------------------
            data, addr = server_socket.recvfrom(65536) # Max packet size boundary
            traffic_state = json.loads(data.decode('utf-8'))
            current_time = traffic_state['Time_sec']

            # Append traffic snap metrics to rolling history window
            live_history_buffer.extend(traffic_state['Vehicles'])
            if len(live_history_buffer) > 2000:
                live_history_buffer = live_history_buffer[-1000:] # Circular trim

            print(f"\n[Tick t={current_time:.2f}s] Packet from {addr[0]} | Vehicles Tracked: {len(traffic_state['Vehicles'])}")

            # Prepare empty structure for response payload back to simulator
            response_payload = {"Commands": []}
            futures = []

            # ------------------------------------------------------------------
            # PHASE 2: THINK (Sequential AI & Concurrent Optimizers)
            # ------------------------------------------------------------------
            # Simulation warmup boundary (16 frames minimum matching args['in_length'])
            if current_time >= 1.6:
                for veh in traffic_state['Vehicles']:
                    ego_id = veh['Car_ID']

                    # Lazy instantiation of tracking state for new entities entering arena
                    if ego_id not in active_agents:
                        active_agents[ego_id] = VehicleControlUnit(car_id=ego_id)

                    # --- Layer 1: CS-LSTM Neural Inference ---
                    try:
                        # Feed rolling tracking history arrays directly into your highwayNet model graph evaluation
                        ego_preds, _ = run_inference(net, ego_id, live_history_buffer, current_time, lane_map)
                    except ValueError:
                        # Skip vehicle processing frame if history length threshold isn't met yet
                        continue

                    # Parse spatial telemetry values for current tracking matrix
                    lat_pos = lane_map.get(str(veh['Lane_ID']), 0.0)
                    current_state = [lat_pos, veh['sigLocalY'], 0.0, veh['sigVel']]
                    final_goal = [current_state[0], current_state[1] + 100.0]

                    # Map environment proximity map matrices (All surrounding targets)
                    nbrs_curr = []
                    for n_veh in traffic_state['Vehicles']:
                        if n_veh['Car_ID'] == ego_id:
                            continue
                        n_lat = lane_map.get(str(n_veh['Lane_ID']), 0.0)
                        nbrs_curr.append([n_lat, n_veh['sigLocalY'], 0.0, n_veh['sigVel']])

                    # --- Layer 2 & 3: Dispatch mathematical operations to the CPU Pool ---
                    future = executor.submit(
                        compute_agent, ego_id, current_state, final_goal,
                        ego_preds, nbrs_curr, [], active_agents[ego_id]
                    )
                    futures.append(future)

                # Reassemble results vector as cores conclude executions
                for future in concurrent.futures.as_completed(futures):
                    ego_id, verdict, success = future.result()
                    if success:
                        response_payload["Commands"].append({
                            "Car_ID": ego_id,
                            "a_x": verdict['u_safe'][0], # Lateral trajectory modification step
                            "a_y": verdict['u_safe'][1]  # Longitudinal acceleration constraint target
                        })

            # ------------------------------------------------------------------
            # PHASE 3: ACT (Return Command Matrix to Linux VM over Network Interface)
            # ------------------------------------------------------------------
            response_json = json.dumps(response_payload)
            server_socket.sendto(response_json.encode('utf-8'), addr)
            print(f"  └─► Echoed execution commands back to VM. Actuated Agents: {len(response_payload['Commands'])}")


if __name__ == "__main__":
    run_realtime_server()
