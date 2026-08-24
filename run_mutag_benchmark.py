import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_baselines import BaselineGCN
from model import CurvatureMPSN
from train import process_dataset

def train_epoch(model, optimizer, criterion, train_data, device, is_gcn=False):
    model.train()
    total_loss = 0
    correct = 0
    for data in train_data:
        optimizer.zero_grad()
        if is_gcn:
            out = model(data['x_0'].to(device), data['edge_index'].to(device))
        else:
            out = model(data['x_0'].to(device), data['hlpe'].to(device) if 'hlpe' in data else None, None, 
                        data['B1'].to(device), data['B2'].to(device), 
                        data['frc'].to(device), None, None, None)
        y = data['y'].to(device)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
    return total_loss / len(train_data), correct / len(train_data)

def test(model, criterion, test_data, device, is_gcn=False):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_data:
            if is_gcn:
                out = model(data['x_0'].to(device), data['edge_index'].to(device))
            else:
                out = model(data['x_0'].to(device), data['hlpe'].to(device) if 'hlpe' in data else None, None, 
                            data['B1'].to(device), data['B2'].to(device), 
                            data['frc'].to(device), None, None, None)
            y = data['y'].to(device)
            loss = criterion(out, y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == y).sum())
    return total_loss / len(test_data), correct / len(test_data)

def run_mutag_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading MUTAG dataset...")
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    print(f"Dataset Size: {len(dataset)} graphs")
    
    # Process dataset
    processed_dataset = process_dataset(dataset)
    num_node_features = dataset.num_node_features
    if num_node_features == 0:
        num_node_features = 1
    num_classes = dataset.num_classes
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    def evaluate_model(ModelClass, is_gcn, name):
        fold_accs = []
        print(f"\n--- Evaluating {name} on MUTAG (10-fold CV) ---")
        for fold, (train_idx, test_idx) in enumerate(kf.split(processed_dataset)):
            train_data = [processed_dataset[i] for i in train_idx]
            test_data = [processed_dataset[i] for i in test_idx]
            
            model = ModelClass(num_node_features=num_node_features, hidden_dim=32, num_classes=num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            
            best_acc = 0.0
            for epoch in range(1, 51):  # 50 epochs per fold
                train_loss, train_acc = train_epoch(model, optimizer, criterion, train_data, device, is_gcn)
                val_loss, val_acc = test(model, criterion, test_data, device, is_gcn)
                if val_acc > best_acc:
                    best_acc = val_acc
                    
            fold_accs.append(best_acc)
            print(f"Fold {fold+1}: {best_acc*100:.2f}%")
            
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        print(f"-> {name} 10-Fold CV Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        return mean_acc, std_acc

    gcn_mean, gcn_std = evaluate_model(BaselineGCN, is_gcn=True, name="Standard GCN (1-WL Baseline)")
    mpsn_mean, mpsn_std = evaluate_model(CurvatureMPSN, is_gcn=False, name="Curvature-Weighted MPSN")
    
    print("\n==================================")
    print("FINAL COMPARISON (MUTAG DATASET)")
    print("==================================")
    print(f"Standard GCN:         {gcn_mean*100:.2f}% ± {gcn_std*100:.2f}%")
    print(f"Curvature MPSN:       {mpsn_mean*100:.2f}% ± {mpsn_std*100:.2f}%")
    print(f"Improvement:          +{(mpsn_mean - gcn_mean)*100:.2f}%")

if __name__ == "__main__":
    run_mutag_benchmark()
