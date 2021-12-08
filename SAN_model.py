"""
    SAN Model Arguments
    ---------
    pnn : peripheral node based neighbors
    pn : peripheral nodes
    kw : keyword
    att: attribute
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv

class pnn_layer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(pnn_layer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        pnn_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            if i != 1:
                new_g = self._cached_coalesced_graph[meta_path]
                pnn_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        pnn_embeddings = torch.stack(pnn_embeddings, dim=1)                  # (N, M, D * K)

        return pnn_embeddings.flatten(1)                           # (N, D * K)

class SAN(nn.Module):
    def __init__(self, meta_paths, in_size, supportFeature_size, hidden_size, out_size, num_heads, dropout):
        super(SAN, self).__init__()

        # aggregator of peripheral nodes based neighbors
        self.layers = nn.ModuleList()
        self.layers.append(pnn_layer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(pnn_layer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))

        # aggregators of peripheral nodes, i.e., query, keyword and attribute
        self.queryAggr = GATConv(64, hidden_size * num_heads[-1], num_heads[0],
                                 dropout, dropout, activation=F.elu,
                                 allow_zero_in_degree=True)
        self.kwAggr = GATConv(64, hidden_size * num_heads[-1], num_heads[0],
                                  dropout, dropout, activation=F.elu,
                                  allow_zero_in_degree=True)
        self.attAggr = GATConv(64, hidden_size * num_heads[-1], num_heads[0],
                              dropout, dropout, activation=F.elu,
                              allow_zero_in_degree=True)

        self.proj_pnn_a = nn.Linear(in_size, 64, bias=False)
        self.proj_pn_a = nn.Linear(supportFeature_size['attIn_size'], 64, bias=False)

        self.proj_pnn = nn.Linear(128, 64, bias=False)
        self.proj_pn = nn.Linear(512, 64, bias=False)
        self.predict = nn.Linear(64*4, out_size)

    def forward(self, g, hp, hq, hk, ha):
        hp = torch.randn(*hp.shape, device = 'cuda')
        hp_cache = hp
        for gnn in self.layers:
            hp = gnn(g, hp)

        queryg = dgl.edge_type_subgraph(g, [('query', 'qp', 'p')])
        bi_queryg = dgl.heterograph({('_U', '_E', '_V') : queryg.edges()})
        hq = self.queryAggr(bi_queryg, (self.proj_pn_a(hq), self.proj_pnn_a(hp_cache)))

        kwg = dgl.edge_type_subgraph(g, [('kw', 'kp', 'p')])
        bi_kwg = dgl.heterograph({('_U', '_E', '_V'): kwg.edges()})
        hk = self.kwAggr(bi_kwg, (self.proj_pn_a(hk), self.proj_pnn_a(hp_cache)))

        attg = dgl.edge_type_subgraph(g, [('att', 'ap', 'p')])
        bi_attg= dgl.heterograph({('_U', '_E', '_V'): attg.edges()})
        ha = self.attAggr(bi_attg, (self.proj_pn_a(ha), self.proj_pnn_a(hp_cache)))

        hp = torch.cat((self.proj_pnn(hp), self.proj_pn(hq.flatten(1)),
                       self.proj_pn(hk.flatten(1)), self.proj_pn(ha.flatten(1))), axis=1)
        return self.predict(hp)