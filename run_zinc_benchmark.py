import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_baselines import BaselineGCN
from model import CurvatureMPSN
from train import process_dataset

def one_hot_features(dataset):
    # ZINC node features are categorical (atom types 0-20)
    # Convert them to one-hot float vectors to be compatible with basic MLPs/GCNs
    for data in dataset:
        if data.x is not None:
            data.x = F.one_hot(data.x.squeeze(-1), num_classes=21).float()
    return dataset

def train_epoch(model, optimizer, criterion, train_data, device, is_gcn=False):
    model.train()
    total_loss = 0
    for data in train_data:
        optimizer.zero_grad()
        if is_gcn:
            out = model(data['x_0'].to(device), data['edge_index'].to(device))
        else:
            out = model(data['x_0'].to(device), data['hlpe'].to(device) if 'hlpe' in data else None, None, 
                        data['B1'].to(device), data['B2'].to(device), 
                        data['frc'].to(device), None, None, None)
        y = data['y'].to(device)
        if out.shape != y.shape:
            out = out.squeeze(-1)
            y = y.squeeze(-1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_data)

def test(model, criterion, test_data, device, is_gcn=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_data:
            if is_gcn:
                out = model(data['x_0'].to(device), data['edge_index'].to(device))
            else:
                out = model(data['x_0'].to(device), data['hlpe'].to(device) if 'hlpe' in data else None, None, 
                            data['B1'].to(device), data['B2'].to(device), 
                            data['frc'].to(device), None, None, None)
            y = data['y'].to(device)
            if out.shape != y.shape:
                out = out.squeeze(-1)
                y = y.squeeze(-1)
            loss = criterion(out, y)
            total_loss += loss.item()
    return total_loss / len(test_data)

def run_zinc_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading ZINC dataset (Subset=True)...")
    train_dataset = ZINC(root='/tmp/ZINC', subset=True, split='train')
    val_dataset = ZINC(root='/tmp/ZINC', subset=True, split='val')
    test_dataset = ZINC(root='/tmp/ZINC', subset=True, split='test')
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # We will use a smaller fraction of train for quick benchmarking (e.g. 500 graphs)
    train_subset = [train_dataset[i] for i in range(500)]
    val_subset = [val_dataset[i] for i in range(100)]
    test_subset = [test_dataset[i] for i in range(100)]
    
    train_subset = one_hot_features(train_subset)
    val_subset = one_hot_features(val_subset)
    test_subset = one_hot_features(test_subset)
    
    print("Lifting Train Set to Cell Complexes...")
    train_proc = process_dataset(train_subset)
    print("Lifting Val Set to Cell Complexes...")
    val_proc = process_dataset(val_subset)
    print("Lifting Test Set to Cell Complexes...")
    test_proc = process_dataset(test_subset)
    
    num_node_features = 21 # one-hot size
    
    def evaluate_model(ModelClass, is_gcn, name):
        print(f"\n--- Evaluating {name} on ZINC ---")
        model = ModelClass(num_node_features=num_node_features, hidden_dim=64, num_classes=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss() # MAE Loss
        
        best_val_mae = float('inf')
        test_mae_at_best_val = float('inf')
        
        for epoch in range(1, 31):
            train_mae = train_epoch(model, optimizer, criterion, train_proc, device, is_gcn)
            val_mae = test(model, criterion, val_proc, device, is_gcn)
            test_mae = test(model, criterion, test_proc, device, is_gcn)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                test_mae_at_best_val = test_mae
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch:02d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")
                
        print(f"-> {name} Final Test MAE: {test_mae_at_best_val:.4f}")
        return test_mae_at_best_val

    gcn_mae = evaluate_model(BaselineGCN, is_gcn=True, name="Standard GCN (1-WL Baseline)")
    mpsn_mae = evaluate_model(CurvatureMPSN, is_gcn=False, name="Curvature-Weighted MPSN")
    
    print("\n==================================")
    print("FINAL COMPARISON (ZINC DATASET)")
    print("==================================")
    print(f"Standard GCN MAE:   {gcn_mae:.4f}")
    print(f"Curvature MPSN MAE: {mpsn_mae:.4f}")
    print(f"Improvement:        {gcn_mae - mpsn_mae:.4f} lower MAE")

if __name__ == "__main__":
    run_zinc_benchmark()
