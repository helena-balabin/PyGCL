from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import dropout_edge


class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_mask = dropout_edge(edge_index, p=self.pe)
        if edge_weights is not None:
            edge_weights = edge_weights[edge_mask]
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
