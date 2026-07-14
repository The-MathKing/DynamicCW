import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import torch_geometric.utils as utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dynamic_cw_network import DynamicCWNetwork, StaticCWNetwork

# ==========================================
# 1. Data Preparation & Transform
# ==========================================
def compute_forman_ricci(data):
    """
    Computes the Augmented combinatorial Forman-Ricci curvature:
    4 - deg(u) - deg(v) + 3 * num_triangles(e) + num_4_cliques(e)
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    deg = utils.degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj_squared = torch.matmul(adj, adj)
    
    u = edge_index[0]
    v = edge_index[1]
    num_triangles = adj_squared[u, v]
    
    num_4_cliques = torch.zeros(edge_index.shape[1], dtype=torch.float)
    for i in range(edge_index.shape[1]):
        ui = edge_index[0, i]
        vi = edge_index[1, i]
        common = torch.where((adj[ui] * adj[vi]) > 0)[0]
        if len(common) >= 2:
            sub_adj = adj[common][:, common]
            num_4_cliques[i] = sub_adj.sum() / 2.0
            
    curvature = 4.0 - deg[u] - deg[v] + 3.0 * num_triangles + num_4_cliques
    data.edge_attr = curvature.view(-1, 1)
    return data

class FormanRicciTransform(object):
    def __call__(self, data):
        return compute_forman_ricci(data)

# ==========================================
# 2. The Baseline Architecture (GINBaseline)
# ==========================================
class GINBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_classes):
        super(GINBaseline, self).__init__()
        self.initial_embedding = nn.Linear(in_channels, hidden_channels)
        self.conv1 = GINConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ELU(), nn.Linear(hidden_channels, hidden_channels)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ELU(), nn.Linear(hidden_channels, hidden_channels)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ELU(), nn.Linear(hidden_channels, hidden_channels)))
        self.mlp_lin1 = nn.Linear(hidden_channels, 16)
        self.dropout = nn.Dropout(p=0.5)
        self.mlp_lin2 = nn.Linear(16, out_classes)
        
    def forward(self, x, edge_index, batch):
        x = F.elu(self.initial_embedding(x))
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.elu(self.mlp_lin1(x))
        x = self.dropout(x)
        x = self.mlp_lin2(x)
        return x

# ==========================================
# 3. The Rigorous Training Crucible (Ablation Edition)
# ==========================================
def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    seeds = [42, 123, 2026, 777, 999]
    num_epochs = 200
    
    all_gin_train_losses = np.zeros((len(seeds), num_epochs))
    all_gin_test_accs = np.zeros((len(seeds), num_epochs))
    
    all_cw_static_train_losses = np.zeros((len(seeds), num_epochs))
    all_cw_static_test_accs = np.zeros((len(seeds), num_epochs))
    
    all_cw_train_losses = np.zeros((len(seeds), num_epochs))
    all_cw_test_accs = np.zeros((len(seeds), num_epochs))
    
    final_metrics = {
        'GIN': {'acc': [], 'f1': [], 'runtime': [], 'cm': []},
        'CW_Static': {'acc': [], 'f1': [], 'runtime': [], 'cm': []},
        'CW_Dynamic': {'acc': [], 'f1': [], 'runtime': [], 'cm': []}
    }
    
    print("Loading NCI1 dataset and calculating Augmented Forman-Ricci curvature...")
    base_dataset = TUDataset(root='./data', name='NCI1', pre_transform=FormanRicciTransform())
    
    in_channels = base_dataset.num_node_features
    out_classes = base_dataset.num_classes
    hidden_channels = 32

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"ANTI-FLUKE PROTOCOL: Starting Seed {seed} ({seed_idx + 1}/{len(seeds)})")
        print(f"{'='*60}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
            
        dataset = base_dataset.shuffle()
        split_idx = int(len(dataset) * 0.8)
        train_dataset = dataset[:split_idx]
        test_dataset = dataset[split_idx:]
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        gin_model = GINBaseline(in_channels, hidden_channels, out_classes).to(device)
        cw_static_model = StaticCWNetwork(in_channels, hidden_channels, out_classes).to(device)
        cw_dyn_model = DynamicCWNetwork(in_channels, hidden_channels, out_classes).to(device)
        
        criterion = nn.CrossEntropyLoss()
        opt_gin = torch.optim.Adam(gin_model.parameters(), lr=0.001)
        opt_cw_static = torch.optim.Adam(cw_static_model.parameters(), lr=0.001)
        opt_cw_dyn = torch.optim.Adam(cw_dyn_model.parameters(), lr=0.001)
        
        gin_runtimes, cw_static_runtimes, cw_dyn_runtimes = [], [], []
        
        for epoch in range(1, num_epochs + 1):
            gin_model.train(); cw_static_model.train(); cw_dyn_model.train()
            
            # GIN
            start_time = time.time()
            total_gin_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                opt_gin.zero_grad()
                out = gin_model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                opt_gin.step()
                total_gin_loss += loss.item() * batch.num_graphs
            gin_runtimes.append(time.time() - start_time)
            
            # Static CW
            start_time = time.time()
            total_static_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                opt_cw_static.zero_grad()
                out = cw_static_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                opt_cw_static.step()
                total_static_loss += loss.item() * batch.num_graphs
            cw_static_runtimes.append(time.time() - start_time)
            
            # Dynamic CW
            start_time = time.time()
            total_dyn_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                opt_cw_dyn.zero_grad()
                out = cw_dyn_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                opt_cw_dyn.step()
                total_dyn_loss += loss.item() * batch.num_graphs
            cw_dyn_runtimes.append(time.time() - start_time)
            
            gin_model.eval(); cw_static_model.eval(); cw_dyn_model.eval()
            gin_corr, cw_static_corr, cw_dyn_corr = 0, 0, 0
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = gin_model(batch.x, batch.edge_index, batch.batch)
                    gin_corr += int((out.argmax(dim=1) == batch.y).sum())
                    
                    out = cw_static_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    cw_static_corr += int((out.argmax(dim=1) == batch.y).sum())
                    
                    out = cw_dyn_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    cw_dyn_corr += int((out.argmax(dim=1) == batch.y).sum())
                    
            gin_acc = (gin_corr / len(test_loader.dataset)) * 100.0
            cw_static_acc = (cw_static_corr / len(test_loader.dataset)) * 100.0
            cw_dyn_acc = (cw_dyn_corr / len(test_loader.dataset)) * 100.0
            
            all_gin_train_losses[seed_idx, epoch - 1] = total_gin_loss / len(train_loader.dataset)
            all_gin_test_accs[seed_idx, epoch - 1] = gin_acc
            all_cw_static_train_losses[seed_idx, epoch - 1] = total_static_loss / len(train_loader.dataset)
            all_cw_static_test_accs[seed_idx, epoch - 1] = cw_static_acc
            all_cw_train_losses[seed_idx, epoch - 1] = total_dyn_loss / len(train_loader.dataset)
            all_cw_test_accs[seed_idx, epoch - 1] = cw_dyn_acc
            
            if epoch % 20 == 0 or epoch == 1:
                print(f"Ep [{epoch}/{num_epochs}] | "
                      f"GIN Acc: {gin_acc:.1f}% | "
                      f"StaticCW Acc: {cw_static_acc:.1f}% | "
                      f"DynCW Acc: {cw_dyn_acc:.1f}%")
        
        # End of seed: Stats
        gin_p, gin_t = [], []
        cw_static_p, cw_static_t = [], []
        cw_dyn_p, cw_dyn_t = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                gin_p.extend(gin_model(batch.x, batch.edge_index, batch.batch).argmax(dim=1).cpu().numpy())
                gin_t.extend(batch.y.cpu().numpy())
                
                cw_static_p.extend(cw_static_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).argmax(dim=1).cpu().numpy())
                cw_static_t.extend(batch.y.cpu().numpy())
                
                cw_dyn_p.extend(cw_dyn_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).argmax(dim=1).cpu().numpy())
                cw_dyn_t.extend(batch.y.cpu().numpy())
                
        # Metrics
        def calc_metrics(preds, targets, runtimes):
            acc = accuracy_score(targets, preds) * 100.0
            f1 = f1_score(targets, preds, average='macro')
            cm = confusion_matrix(targets, preds, labels=[0, 1])
            rt = np.mean(runtimes)
            return acc, f1, cm, rt
            
        gin_acc, gin_f1, gin_cm, gin_rt = calc_metrics(gin_p, gin_t, gin_runtimes)
        stat_acc, stat_f1, stat_cm, stat_rt = calc_metrics(cw_static_p, cw_static_t, cw_static_runtimes)
        dyn_acc, dyn_f1, dyn_cm, dyn_rt = calc_metrics(cw_dyn_p, cw_dyn_t, cw_dyn_runtimes)
        
        final_metrics['GIN']['acc'].append(gin_acc)
        final_metrics['GIN']['f1'].append(gin_f1)
        final_metrics['GIN']['cm'].append(gin_cm.tolist())
        final_metrics['GIN']['runtime'].append(gin_rt)
        
        final_metrics['CW_Static']['acc'].append(stat_acc)
        final_metrics['CW_Static']['f1'].append(stat_f1)
        final_metrics['CW_Static']['cm'].append(stat_cm.tolist())
        final_metrics['CW_Static']['runtime'].append(stat_rt)
        
        final_metrics['CW_Dynamic']['acc'].append(dyn_acc)
        final_metrics['CW_Dynamic']['f1'].append(dyn_f1)
        final_metrics['CW_Dynamic']['cm'].append(dyn_cm.tolist())
        final_metrics['CW_Dynamic']['runtime'].append(dyn_rt)
        
        print(f"\n--- Seed {seed} Final Stats ---")
        print(f"GINBaseline   -> Acc: {gin_acc:.2f}%, F1: {gin_f1:.4f}")
        print(f"StaticCWNet   -> Acc: {stat_acc:.2f}%, F1: {stat_f1:.4f}")
        print(f"DynamicCWNet  -> Acc: {dyn_acc:.2f}%, F1: {dyn_f1:.4f}")

    # Export
    epoch_df = pd.DataFrame({
        'Epoch': np.arange(1, num_epochs + 1),
        'GIN_Train_Loss_Mean': np.mean(all_gin_train_losses, axis=0),
        'GIN_Test_Acc_Mean': np.mean(all_gin_test_accs, axis=0),
        'StaticCW_Train_Loss_Mean': np.mean(all_cw_static_train_losses, axis=0),
        'StaticCW_Test_Acc_Mean': np.mean(all_cw_static_test_accs, axis=0),
        'DynamicCW_Train_Loss_Mean': np.mean(all_cw_train_losses, axis=0),
        'DynamicCW_Test_Acc_Mean': np.mean(all_cw_test_accs, axis=0)
    })
    epoch_df.to_csv('ablation_loss_curves.csv', index=False)
    
    def build_stats_dict(model_key):
        arr = final_metrics[model_key]
        cm_avg = np.mean(arr['cm'], axis=0)
        tn, fp, fn, tp = cm_avg.ravel()
        return {
            'Accuracy': {'Mean': float(np.mean(arr['acc'])), 'Std_Dev': float(np.std(arr['acc']))},
            'F1_Macro': {'Mean': float(np.mean(arr['f1'])), 'Std_Dev': float(np.std(arr['f1']))},
            'Average_Runtime_Per_Epoch_Sec': {'Mean': float(np.mean(arr['runtime'])), 'Std_Dev': float(np.std(arr['runtime']))}
        }

    final_stats_json = {
        'GINBaseline': build_stats_dict('GIN'),
        'StaticCWNetwork': build_stats_dict('CW_Static'),
        'DynamicCWNetwork': build_stats_dict('CW_Dynamic')
    }
    
    with open('ablation_statistics.json', 'w') as f:
        json.dump(final_stats_json, f, indent=4)
        
    print("\nAblation study completed. Exported ablation_loss_curves.csv and ablation_statistics.json.")

if __name__ == "__main__":
    main()
