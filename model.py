"""
DynamicCW and Cellular Message Passing Architecture.
Implements:
- CurvatureWeightedCellularConv with dynamic 2-cell updates, residual connections, and layer normalization.
- Curvature gating options (vector, scalar, none) with standardized curvature representations.
- Permutation-equivariant readout supporting 'sum', 'mean', and hybrid 'sum_mean' global pooling.
- Ablation toggles: static vs dynamic 2-cells, with/without residuals, with/without curvature gates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CellularMLP(nn.Module):
    def __init__(self, in_channels, out_channels, use_eps=True):
        super(CellularMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        self.use_eps = use_eps
        if use_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('eps', torch.tensor(0.0))
            
    def forward(self, x, messages):
        if x is None:
            return self.net(messages)
        if self.use_eps:
            return self.net((1.0 + self.eps) * x + messages)
        return self.net(x + messages)

class CurvatureWeightedCellularConv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        gating='vector', 
        dynamic_faces=True,
        use_residuals=True,
        use_norm=True
    ):
        super(CurvatureWeightedCellularConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gating = gating
        self.dynamic_faces = dynamic_faces
        self.use_residuals = use_residuals
        self.use_norm = use_norm
        
        self.mlp_node = CellularMLP(in_channels, out_channels)
        self.mlp_edge = CellularMLP(in_channels, out_channels)
        if self.dynamic_faces:
            self.mlp_face = CellularMLP(in_channels, out_channels)
            
        self.lin_down = nn.Linear(in_channels, in_channels)
        self.lin_up = nn.Linear(in_channels, in_channels)
        self.lin_adj_down = nn.Linear(in_channels, in_channels)
        self.lin_adj_up = nn.Linear(in_channels, in_channels)
        self.lin_edge_to_node = nn.Linear(in_channels, in_channels)
        self.lin_edge_to_face = nn.Linear(in_channels, in_channels)
        
        if self.gating == 'scalar':
            self.gate_proj = nn.Linear(in_channels + 1, 1)
        elif self.gating == 'vector':
            self.gate_proj = nn.Linear(in_channels + 1, out_channels)
            
        if self.use_norm:
            self.norm_node = nn.LayerNorm(out_channels)
            self.norm_edge = nn.LayerNorm(out_channels)
            if self.dynamic_faces:
                self.norm_face = nn.LayerNorm(out_channels)

    def forward(self, x_0, x_1, x_2, incidence_1, incidence_2, frc_weights):
        """
        x_0: 0-cells (nodes) [N_0, D]
        x_1: 1-cells (edges) [N_1, D]
        x_2: 2-cells (faces) [N_2, D]
        incidence_1: |B1| (|V| x |E|) absolute boundary matrix
        incidence_2: |B2| (|E| x |F|) absolute boundary matrix
        frc_weights: [N_1, 1] discrete curvature vector
        """
        # Ensure absolute boundary values
        if incidence_1 is not None:
            if incidence_1.is_sparse:
                inc_1 = torch.sparse_coo_tensor(
                    incidence_1._indices(), torch.abs(incidence_1._values()), incidence_1.shape
                ).coalesce()
            else:
                inc_1 = torch.abs(incidence_1)
        else:
            inc_1 = None
            
        if incidence_2 is not None and incidence_2.shape[1] > 0:
            if incidence_2.is_sparse:
                inc_2 = torch.sparse_coo_tensor(
                    incidence_2._indices(), torch.abs(incidence_2._values()), incidence_2.shape
                ).coalesce()
            else:
                inc_2 = torch.abs(incidence_2)
        else:
            inc_2 = None

        # Standardize curvature per graph
        if frc_weights is not None and frc_weights.shape[0] > 1:
            std = frc_weights.std()
            if std > 1e-6:
                frc_norm = (frc_weights - frc_weights.mean()) / (std + 1e-5)
            else:
                frc_norm = frc_weights - frc_weights.mean()
        else:
            frc_norm = frc_weights if frc_weights is not None else torch.zeros((x_1.shape[0], 1), device=x_1.device)
            
        # 1. Edge Message Aggregation
        msg_up = 0.0
        if x_0 is not None and inc_1 is not None and inc_1.shape[1] > 0:
            x_0_t = self.lin_down(x_0)
            msg_up = torch.sparse.mm(inc_1.t(), x_0_t) if inc_1.is_sparse else torch.matmul(inc_1.t(), x_0_t)
            
        msg_down = 0.0
        if x_2 is not None and inc_2 is not None and inc_2.shape[1] > 0 and x_2.shape[0] > 0:
            x_2_t = self.lin_up(x_2)
            msg_down = torch.sparse.mm(inc_2, x_2_t) if inc_2.is_sparse else torch.matmul(inc_2, x_2_t)
            
        msg_adj_down = 0.0
        if inc_1 is not None and inc_1.shape[1] > 0:
            x_1_adj_d = self.lin_adj_down(x_1)
            temp = torch.sparse.mm(inc_1, x_1_adj_d) if inc_1.is_sparse else torch.matmul(inc_1, x_1_adj_d)
            msg_adj_down = torch.sparse.mm(inc_1.t(), temp) if inc_1.is_sparse else torch.matmul(inc_1.t(), temp)
            
        msg_adj_up = 0.0
        if inc_2 is not None and inc_2.shape[1] > 0:
            x_1_adj_u = self.lin_adj_up(x_1)
            temp = torch.sparse.mm(inc_2.t(), x_1_adj_u) if inc_2.is_sparse else torch.matmul(inc_2.t(), x_1_adj_u)
            msg_adj_up = torch.sparse.mm(inc_2, temp) if inc_2.is_sparse else torch.matmul(inc_2, temp)
            
        msg_edge = msg_up + msg_down + msg_adj_down + msg_adj_up
        edge_raw = self.mlp_edge(x_1, msg_edge)
        
        # Curvature gating
        if self.gating in ['scalar', 'vector']:
            gate_in = torch.cat([x_1, frc_norm], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_in))
            edge_update = edge_raw * gate
        else:
            edge_update = edge_raw
            
        if self.use_residuals and x_1.shape == edge_update.shape:
            x_1_new = x_1 + F.gelu(edge_update)
        else:
            x_1_new = F.gelu(edge_update)
            
        if self.use_norm:
            x_1_new = self.norm_edge(x_1_new)
            
        # 2. Node Update
        if x_0 is not None and inc_1 is not None and inc_1.shape[1] > 0:
            msg_e_to_n = torch.sparse.mm(inc_1, x_1) if inc_1.is_sparse else torch.matmul(inc_1, x_1)
            msg_node = self.lin_edge_to_node(msg_e_to_n)
            node_raw = self.mlp_node(x_0, msg_node)
            if self.use_residuals and x_0.shape == node_raw.shape:
                x_0_new = x_0 + F.gelu(node_raw)
            else:
                x_0_new = F.gelu(node_raw)
            if self.use_norm:
                x_0_new = self.norm_node(x_0_new)
        else:
            x_0_new = x_0
            
        # 3. Face Update
        if self.dynamic_faces and x_2 is not None and inc_2 is not None and inc_2.shape[1] > 0 and x_2.shape[0] > 0:
            msg_e_to_f = torch.sparse.mm(inc_2.t(), x_1) if inc_2.is_sparse else torch.matmul(inc_2.t(), x_1)
            msg_face = self.lin_edge_to_face(msg_e_to_f)
            face_raw = self.mlp_face(x_2, msg_face)
            if self.use_residuals and x_2.shape == face_raw.shape:
                x_2_new = x_2 + F.gelu(face_raw)
            else:
                x_2_new = F.gelu(face_raw)
            if self.use_norm:
                x_2_new = self.norm_face(x_2_new)
        else:
            x_2_new = x_2
            
        return x_0_new, x_1_new, x_2_new

class DynamicCWNet(nn.Module):
    def __init__(
        self,
        num_node_features,
        hidden_dim=64,
        num_classes=1,
        num_layers=2,
        gating='vector',
        readout='sum',
        dynamic_faces=True,
        use_residuals=True,
        use_norm=True,
        edge_feature_dim=8,
        face_feature_dim=1,
        task_type='regression'
    ):
        super(DynamicCWNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.readout = readout
        self.dynamic_faces = dynamic_faces
        self.task_type = task_type
        
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim)
        self.face_embedding = nn.Linear(face_feature_dim, hidden_dim)
        
        self.convs = nn.ModuleList([
            CurvatureWeightedCellularConv(
                hidden_dim, 
                hidden_dim, 
                gating=gating, 
                dynamic_faces=dynamic_faces,
                use_residuals=use_residuals,
                use_norm=use_norm
            )
            for _ in range(num_layers)
        ])
        
        readout_mult = 3 if dynamic_faces else 2
        if readout == 'sum_mean':
            readout_mult *= 2
            
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * readout_mult, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def _pool(self, x, batch):
        if x is None or x.shape[0] == 0:
            return torch.zeros((1, self.hidden_dim), device=self.classifier[0].weight.device)
            
        if batch is not None:
            from torch_geometric.nn import global_mean_pool, global_add_pool
            if self.readout == 'sum':
                return global_add_pool(x, batch)
            elif self.readout == 'mean':
                return global_mean_pool(x, batch)
            elif self.readout == 'sum_mean':
                return torch.cat([global_add_pool(x, batch), global_mean_pool(x, batch)], dim=-1)
        else:
            if self.readout == 'sum':
                return torch.sum(x, dim=0, keepdim=True)
            elif self.readout == 'mean':
                return torch.mean(x, dim=0, keepdim=True)
            elif self.readout == 'sum_mean':
                return torch.cat([torch.sum(x, dim=0, keepdim=True), torch.mean(x, dim=0, keepdim=True)], dim=-1)

    def forward(self, x_0, x_1, x_2, incidence_1, incidence_2, frc_weights, batch_0=None, batch_1=None, batch_2=None):
        # Embed 0-cells
        x_0 = self.node_embedding(x_0)
        
        # Embed 1-cells
        if x_1 is None:
            num_edges = incidence_1.shape[1] if incidence_1 is not None else 0
            x_1 = torch.zeros((num_edges, self.edge_embedding.in_features), device=x_0.device)
        x_1 = self.edge_embedding(x_1)
        
        # Embed 2-cells
        if x_2 is None:
            num_faces = incidence_2.shape[1] if (incidence_2 is not None and incidence_2.shape[1] > 0) else 0
            x_2 = torch.ones((num_faces, self.face_embedding.in_features), device=x_0.device)
        x_2 = self.face_embedding(x_2)
        
        for conv in self.convs:
            x_0, x_1, x_2 = conv(x_0, x_1, x_2, incidence_1, incidence_2, frc_weights)
            
        pooled_0 = self._pool(x_0, batch_0)
        pooled_1 = self._pool(x_1, batch_1)
        
        if self.dynamic_faces:
            pooled_2 = self._pool(x_2, batch_2)
            graph_embedding = torch.cat([pooled_0, pooled_1, pooled_2], dim=-1)
        else:
            graph_embedding = torch.cat([pooled_0, pooled_1], dim=-1)
            
        out = self.classifier(graph_embedding)
        return out

# Alias for backward compatibility
CurvatureMPSN = DynamicCWNet
