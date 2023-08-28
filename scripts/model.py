import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList, ReLU, Dropout
import torch.nn as nn
from torch_geometric.nn import TransformerConv, TopKPooling, GCNConv, GATv2Conv, GraphNorm, GlobalAttention, GATConv
# from torch_geometric.nn import global_mean_pool as gap


class GCN(nn.Module):
    def __init__(self, feature_dim, embedding_dim, linear_dim, output_dim, num_layers, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature_dim, embedding_dim)
        self.gn1 = GraphNorm(embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)
        self.gn2 = GraphNorm(embedding_dim)
        # self.pool = TopKPooling(embedding_dim, ratio=0.2)
        self.att = GlobalAttention(nn.Sequential(Linear(embedding_dim, 2*embedding_dim),\
                                                       BatchNorm1d(2*embedding_dim),\
                                                        ReLU(),\
                                                        Linear(2*embedding_dim, 1)))
        self.d_out = dropout_rate


        layers = []
        prev_dim = embedding_dim
        for i in range(num_layers):
            layers.append(Linear(prev_dim, linear_dim))
            layers.append(BatchNorm1d(linear_dim))
            layers.append(ReLU())
            layers.append(Dropout(p=self.d_out))
            prev_dim = linear_dim

        # append last layer
        layers.append(Linear(linear_dim, output_dim))
        self.linear_layers = nn.Sequential(*layers)

        # self.linear_layers = ModuleList([])
        # self.linear_layers.append(Linear(embedding_dim, linear_dim))
        
        # prev_dim = embedding_dim
        # for i in range(num_layers):
        #     self.linear_layers.append(Linear(prev_dim, linear_dim))
        #     prev_dim = linear_dim
        
        # self.linear_layers.append(Linear(linear_dim, output_dim))


        
        
        # self.init_weights()

    # init weights for linear layers
    def init_weights(self):
        for layer in self.linear_layers:
            if isinstance(layer, nn.Linear) and layer.out_features > 2:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    
    def forward(self, x, edge_attr, edge_index, batch):
        
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.gn1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=self.d_out, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.gn2(x)
        x = F.relu(x)
        # x = F.dropout(x, p=self.d_out, training=self.training)
        x= self.att(x, batch)
        # x , edge_index, edge_attr, batch_index, _, _ = self.pool(x, edge_index, edge_attr, batch)
        # x = gap(x, batch)
        # for i in range(len(self.linear_layers)-1):
        #     x = self.linear_layers[i](x)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.d_out, training=self.training)
        
        # x = self.linear_layers[-1](x)
        x = self.linear_layers(x)
        return x

class GAT(torch.nn.Module):
    def __init__(self, feature_dim, embedding_dim, linear_dim, output_dim, num_layers, dropout_rate, heads=4):
        super(GAT, self).__init__()
        
        self.conv1 = GATConv(feature_dim, embedding_dim, heads=4)  # You can adjust the number of heads
        self.gn1 = GraphNorm(embedding_dim*heads)
        self.conv2 = GATConv(embedding_dim * heads, embedding_dim, heads=4)  # Input dimension is multiplied by the number of heads from previous layer
        self.gn2 = GraphNorm(embedding_dim*heads)
        
        self.att = GlobalAttention(torch.nn.Sequential(torch.nn.Linear(embedding_dim*heads, 2*embedding_dim),\
                                                       torch.nn.BatchNorm1d(2*embedding_dim),\
                                                       torch.nn.ReLU(),\
                                                       torch.nn.Linear(2*embedding_dim, 1)))
        self.d_out = dropout_rate
        layers = []
        prev_dim = embedding_dim*heads
        for i in range(num_layers):
            layers.append(Linear(prev_dim, linear_dim))
            layers.append(BatchNorm1d(linear_dim))
            layers.append(ReLU())
            layers.append(Dropout(p=self.d_out))
            prev_dim = linear_dim

        # append last layer
        layers.append(Linear(linear_dim, output_dim))
        self.linear_layers = nn.Sequential(*layers)
        
        # self.init_weights()
    
    # Init weights for linear layers
    def init_weights(self):
        for layer in self.linear_layers:
            if isinstance(layer, nn.Linear) and layer.out_features > 2:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, edge_attr, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.gn2(x)
        x = F.relu(x)
        x = self.att(x, batch)
        # for i in range(len(self.linear_layers)-1):
        #     x = self.linear_layers[i](x)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.d_out, training=self.training)
        
        # x = self.linear_layers[-1](x)
        x = self.linear_layers(x)
        return x

class GAT2(torch.nn.Module):
    def __init__(self, feature_dim, embedding_dim, linear_dim, output_dim, num_layers, dropout_rate, heads=4):
        super(GAT2, self).__init__()
        
        self.conv1 = GATv2Conv(feature_dim, embedding_dim, heads=heads, edge_dim=1)  # Use GATv2Conv
        self.gn1 = GraphNorm(embedding_dim*heads)
        self.conv2 = GATv2Conv(embedding_dim * heads, embedding_dim, heads=heads, edge_dim=1)  # Use GATv2Conv
        self.gn2 = GraphNorm(embedding_dim*heads)
        
        self.att = GlobalAttention(torch.nn.Sequential(torch.nn.Linear(embedding_dim*heads, 2*embedding_dim, bias=False),\
                                                       torch.nn.BatchNorm1d(2*embedding_dim),\
                                                       torch.nn.ReLU(),\
                                                       torch.nn.Linear(2*embedding_dim, 1)))
        
        self.d_out = dropout_rate
        layers = []
        prev_dim = embedding_dim*heads
        for i in range(num_layers):
            layers.append(Linear(prev_dim, linear_dim, bias=False))
            layers.append(BatchNorm1d(linear_dim))
            # layers.append(ReLU())
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            layers.append(Dropout(p=self.d_out))
            prev_dim = linear_dim

        # append last layer
        layers.append(Linear(linear_dim, output_dim))
        self.linear_layers = nn.Sequential(*layers)
        # self.init_weights()
    
    # Init weights for linear layers
    def init_weights(self):
        for layer in self.linear_layers:
            if isinstance(layer, nn.Linear) and layer.out_features > 2:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, edge_attr, edge_index, batch):
        x = self.conv1(x, edge_index, edge_attr)  # Pass edge_attr to the first GATv2Conv layer
        x = self.gn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)  # Pass edge_attr to the second GATv2Conv layer
        x = self.gn2(x)
        x = F.relu(x)
        x = self.att(x, batch)
        # for i in range(len(self.linear_layers)-1):
        #     x = self.linear_layers[i](x)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.d_out, training=self.training)
        
        # x = self.linear_layers[-1](x)
        x = self.linear_layers(x)
        return x

# class GNN(torch.nn.Module):
#     def __init__(self, feature_size, model_params):
#         super(GNN, self).__init__()
#         embedding_size = model_params["model_embedding_size"]
#         n_heads = model_params["model_attention_heads"]
#         self.n_layers = model_params["model_layers"]
#         dropout_rate = model_params["model_dropout_rate"]
#         top_k_ratio = model_params["model_top_k_ratio"]
#         self.top_k_every_n = model_params["model_top_k_every_n"]
#         dense_neurons = model_params["model_dense_neurons"]
#         edge_dim = model_params["model_edge_dim"]

#         self.conv_layers = ModuleList([])
#         self.transf_layers = ModuleList([])
#         self.pooling_layers = ModuleList([])
#         self.bn_layers = ModuleList([])

#         # Transformation layer
#         self.conv1 = TransformerConv(feature_size, 
#                                     embedding_size, 
#                                     heads=n_heads, 
#                                     dropout=dropout_rate,
#                                     edge_dim=edge_dim,
#                                     beta=True) 

#         self.transf1 = Linear(embedding_size*n_heads, embedding_size)
#         self.bn1 = BatchNorm1d(embedding_size)

#         # Other layers
#         for i in range(self.n_layers):
#             self.conv_layers.append(TransformerConv(embedding_size, 
#                                                     embedding_size, 
#                                                     heads=n_heads, 
#                                                     dropout=dropout_rate,
#                                                     edge_dim=edge_dim,
#                                                     beta=True))

#             self.transf_layers.append(Linear(embedding_size*n_heads, embedding_size))
#             self.bn_layers.append(BatchNorm1d(embedding_size))
#             if i % self.top_k_every_n == 0:
#                 self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))
            

#         # Linear layers
#         self.linear1 = Linear(embedding_size*2, dense_neurons)
#         self.linear2 = Linear(dense_neurons, int(dense_neurons/2))  
#         self.linear3 = Linear(int(dense_neurons/2), 1)  

#     def forward(self, x, edge_attr, edge_index, batch_index):
#         # Initial transformation
#         x = self.conv1(x, edge_index, edge_attr)
#         x = torch.relu(self.transf1(x))
#         x = self.bn1(x)

#         # Holds the intermediate graph representations
#         global_representation = []

#         for i in range(self.n_layers):
#             x = self.conv_layers[i](x, edge_index, edge_attr)
#             x = torch.relu(self.transf_layers[i](x))
#             x = self.bn_layers[i](x)
#             # Always aggregate last layer
#             if i % self.top_k_every_n == 0 or i == self.n_layers:
#                 x , edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)](
#                     x, edge_index, edge_attr, batch_index
#                     )
#                 # Add current representation
#                 global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
    
#         x = sum(global_representation)

#         # Output block
#         x = torch.relu(self.linear1(x))
#         x = F.dropout(x, p=0.8, training=self.training)
#         x = torch.relu(self.linear2(x))
#         x = F.dropout(x, p=0.8, training=self.training)
#         x = self.linear3(x)

#         return x