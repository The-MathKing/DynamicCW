import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class BaselineGCN(nn.Module):
    """
    Standard 1-WL Graph Convolutional Network (GCN).
    Used as the mathematical baseline to prove the expressivity failure
    on Strongly Regular graphs, and as a performance baseline on MANTRA/MUTAG.
    """
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(BaselineGCN, self).__init__()
        
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, batch=None):
        # 1-WL Message Passing
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        
        # Global Pooling (Readout)
        if batch is None:
            pooled = torch.mean(x, dim=0, keepdim=True)
        else:
            pooled = global_mean_pool(x, batch)
            
        # Classification
        out = self.classifier(pooled)
        return out
