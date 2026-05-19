from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedNLL, maskedMSETest, maskedNLLTest
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np

args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = True
args['train_flag'] = False

# Initialize network
net = highwayNet(args)
try:
    net.load_state_dict(torch.load('trained_models/cslstm_m.tar'))
except:
    print("Warning: Model file not found. Check the path.")

if args['use_cuda']:
    net = net.cuda()

net.eval() 

tsSet = ngsimDataset('data/TestSet.mat')
tsDataloader = DataLoader(tsSet, batch_size=64, shuffle=False, num_workers=8, collate_fn=tsSet.collate_fn)

results = []

print("Starting Multi-Agent Evaluation...")

with torch.no_grad():
    for i, data in enumerate(tsDataloader):
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

        if args['use_cuda']:
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = \
                hist.cuda(), nbrs.cuda(), mask.cuda(), lat_enc.cuda(), lon_enc.cuda(), fut.cuda(), op_mask.cuda()

        fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
        
        lat_probs = torch.softmax(lat_pred, dim=1) 
        lon_probs = torch.softmax(lon_pred, dim=1)

        num_agents = hist.shape[1]
        for v_idx in range(num_agents):
            
            maneuver_probs = []
            for k in range(args['num_lon_classes']):
                for l in range(args['num_lat_classes']):
                    p = (lat_probs[v_idx, l] * lon_probs[v_idx, k]).item()
                    maneuver_probs.append(p)
            
            best_m = np.argmax(maneuver_probs)
            best_traj = fut_pred[best_m][:, v_idx, :].cpu().numpy()
            
            row = {
                'Frame_ID': i,
                'Vehicle_Index': v_idx,
                'is_ego': 1 if v_idx == 0 else 0,
                'Prob_Stay_Straight': maneuver_probs[0],
                'Prob_Right_Turn':    maneuver_probs[1],
                'Prob_Left_Turn':     maneuver_probs[2],
                'Prob_Brake_Straight':maneuver_probs[3],
                'Prob_Brake_Right':   maneuver_probs[4],
                'Prob_Brake_Left':    maneuver_probs[5],
                'X_Future':   best_traj[:, 0].tolist(),
                'Y_Future':   best_traj[:, 1].tolist(),
                'Sigma_X':    np.exp(best_traj[:, 2]).tolist(),
                'Sigma_Y':    np.exp(best_traj[:, 3]).tolist()
            }
            results.append(row)

        if i % 50 == 0:
            print(f"Processed {i} frames...")
        
        if i >= 300: break

print("Saving results to CSV...")
df = pd.DataFrame(results)
df.to_csv('grad_project_predictions.csv', index=False)
print("'grad_project_predictions.csv' is ready for MPC layer.")