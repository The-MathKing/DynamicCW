import networkx as nx
import toponetx as tnx

def test_cell_attributes():
    G = nx.Graph()
    G.add_edges_from([(1,2), (2,3), (3,4), (4,1)])
    cc = tnx.CellComplex(G)
    
    for edge in cc.skeleton(1):
        cc.set_cell_attributes({tuple(edge): 0.5}, name='frc', rank=1)
        
    print(cc.get_cell_attributes('frc', rank=1))

if __name__ == "__main__":
    test_cell_attributes()
