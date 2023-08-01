import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GCN(torch.nn.Module):
    def __init__(self, feature_dim, embedding_dim, linear_dim, output_dim, num_layers, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature_dim, embedding_dim)
        # self.bn1 = BatchNorm1d(embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)
        # self.bn2 = BatchNorm1d(embedding_dim)
        self.linear_layers = ModuleList([])
        self.linear_layers.append(Linear(embedding_dim, linear_dim))
        for i in range(1, num_layers-1):
            self.linear_layers.append(Linear(linear_dim, linear_dim))

        self.linear_layers.append(Linear(linear_dim, output_dim))
        self.d_out = dropout_rate
        print(len(self.linear_layers))

    def forward(self, x, edge_attr, edge_index, batch):
        
        

        # # for each graph in batch, standardize edge weights
        # for i in range(len(batch)):
        #     # get the start and end index of the graph
        #     start_idx = batch[i]
        #     if i == len(batch)-1:
        #         end_idx = edge_index.shape[1]
        #     else:
        #         end_idx = batch[i+1]
        #     # get the edge weights
        #     edge_weights = edge_attr[start_idx:end_idx]
        #     # standardize the edge weights
        #     edge_weights = (edge_weights - torch.mean(edge_weights))/torch.std(edge_weights)
        #     # set the edge weights
        #     edge_attr[start_idx:end_idx] = edge_weights



        x = self.conv1(x, edge_index, edge_attr)
        # x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.d_out, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        # x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.d_out, training=self.training)
        x = gap(x, batch)

        for i in range(len(self.linear_layers)-1):
            x = self.linear_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.d_out, training=self.training)
        
        x = self.linear_layers[-1](x)
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