import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
import numpy as np
import time

from data_processing import lift_graph_to_simplicial_complex
from model import CurvatureMPSN

def get_incidence_matrices(sc):
    """
    Extracts the B1 (nodes to edges) and B2 (edges to triangles) 
    incidence matrices from a TopoNetX SimplicialComplex as sparse PyTorch tensors.
    """
    # toponetx incidence matrices are scipy sparse matrices
    if sc.dim >= 1:
        B1_scipy = sc.incidence_matrix(rank=1, signed=True)
        # Convert scipy sparse to pytorch sparse
        coo = B1_scipy.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = coo.shape
        B1 = torch.sparse_coo_tensor(indices, values, size=shape)
    else:
        B1 = torch.zeros((len(sc.skeleton(0)), 0))
        
    if sc.dim >= 2:
        B2_scipy = sc.incidence_matrix(rank=2, signed=True)
        coo = B2_scipy.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = coo.shape
        B2 = torch.sparse_coo_tensor(indices, values, size=shape)
    else:
        B2 = torch.zeros((len(sc.skeleton(1)), 0))
        
    return B1, B2

def process_dataset(dataset):
    """
    Preprocess all graphs in the dataset into their topological representations.
    """
    processed_data = []
    print("Lifting graphs to Simplicial Complexes...")
    for i, data in enumerate(dataset):
        # Lift graph
        sc, _ = lift_graph_to_simplicial_complex(data)
        
        # Get incidence matrices
        B1, B2 = get_incidence_matrices(sc)
        
        # Get node features if they exist, otherwise use degree or constant
        if hasattr(data, 'x') and data.x is not None:
            x_0 = data.x
        else:
            # Use constant feature if node features are missing
            x_0 = torch.ones((len(sc.skeleton(0)), 1))
            
        # Get edge curvatures
        if sc.dim >= 1:
            frc_dict = sc.get_simplex_attributes('frc')
            # TopoNetX keeps simplices in order of insertion, but we ensure we extract them properly
            frc_list = [frc_dict[tuple(edge)] for edge in sc.skeleton(1)]
            frc_weights = torch.tensor(frc_list, dtype=torch.float32).unsqueeze(1)
        else:
            frc_weights = torch.empty((0, 1))
            
        # Graph label
        y = data.y
        
        processed_data.append({
            'x_0': x_0,
            'B1': B1,
            'B2': B2,
            'frc_weights': frc_weights,
            'y': y
        })
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(dataset)} graphs")
            
    return processed_data

def train_epoch(model, optimizer, criterion, train_data, device):
    model.train()
    total_loss = 0
    correct = 0
    
    for data in train_data:
        x_0 = data['x_0'].to(device)
        B1 = data['B1'].to(device)
        B2 = data['B2'].to(device)
        frc = data['frc_weights'].to(device)
        y = data['y'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (passing None for batched indices as we process batch size 1 here)
        out = model(x_0, None, None, B1, B2, frc, None, None, None)
        
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
        
    return total_loss / len(train_data), correct / len(train_data)

def test(model, criterion, test_data, device):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data in test_data:
            x_0 = data['x_0'].to(device)
            B1 = data['B1'].to(device)
            B2 = data['B2'].to(device)
            frc = data['frc_weights'].to(device)
            y = data['y'].to(device)
            
            out = model(x_0, None, None, B1, B2, frc, None, None, None)
            loss = criterion(out, y)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == y).sum())
            
    return total_loss / len(test_data), correct / len(test_data)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading MUTAG Dataset...")
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    
    # We define number of node features and classes
    num_node_features = dataset.num_node_features if dataset.num_node_features > 0 else 1
    num_classes = dataset.num_classes
    
    # Process dataset
    processed_dataset = process_dataset(dataset)
    
    # 10-Fold Cross Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    all_test_accs = []
    epoch_runtimes = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(processed_dataset)):
        print(f"--- Fold {fold+1}/10 ---")
        
        train_data = [processed_dataset[i] for i in train_idx]
        test_data = [processed_dataset[i] for i in test_idx]
        
        # Initialize model
        model = CurvatureMPSN(num_node_features=num_node_features, hidden_dim=32, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Train for 20 epochs for demonstration
        for epoch in range(1, 21):
            start_time = time.time()
            train_loss, train_acc = train_epoch(model, optimizer, criterion, train_data, device)
            end_time = time.time()
            
            epoch_time = end_time - start_time
            epoch_runtimes.append(epoch_time)
            
            test_loss, test_acc = test(model, criterion, test_data, device)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Time: {epoch_time:.3f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
                
        all_test_accs.append(test_acc)
        
    print(f"\nFinal 10-Fold Results:")
    print(f"Accuracy: {np.mean(all_test_accs)*100:.2f}% ± {np.std(all_test_accs)*100:.2f}%")
    print(f"Average Runtime per Epoch: {np.mean(epoch_runtimes):.3f}s ± {np.std(epoch_runtimes):.3f}s")

