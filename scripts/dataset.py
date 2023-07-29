from torch_geometric.data import Data, Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import os.path as osp
import torch
from utils import *

class FinlandDataset(Dataset):
    def __init__(self, root, k="none",  transform=None, pre_transform=None):
        """
        root: root directory of the dataset, folder is spliut into raw_dir and processed_dir
        transform: transform to apply to each data instance
        pre_transform: transform to apply to the whole dataset
        """
        self.k = k
        self.map = []
        super(FinlandDataset, self).__init__(root, transform, pre_transform)


    @property
    def raw_file_names(self):
        """
        returns the names of the raw files in the raw_dir
        """
        return ["ap_coords.csv", "scans.csv", "scan_coords.csv"]
    
    @property
    def processed_file_names(self):
        """
        returns the names of the processed files in the processed_dir
        """
        data = pd.read_csv(osp.join(self.raw_paths[1]))
        return [f"data_{i}.pt" for i in range(len(data))]
    
    def download(self):
        """
        downloads the dataset from the internet
        """
        pass
    
    def _convNxGraphToPyGData(self, ap_coords, scan, scan_coords):

        """ Helper function to convert a networkx graph to a PyG data object. """

        # get graph
        G, relative_scan_coords = create_graph_from_scan(ap_coords, scan, scan_coords, k=self.k)

        if not self.test:
            self.map_train.append((scan_coords, relative_scan_coords))
        else:
            self.map_test.append((scan_coords, relative_scan_coords))
        # convert node labels to integers
        G = nx.convert_node_labels_to_integers(G)

        # check if graph is direced or not, if yes convert to directed
        G = G.to_directed() if not nx.is_directed(G) else G

        # get edge index
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        # get node attributes
        node_attrs = []
        for node in G.nodes:
            node_attrs.append(list(G.nodes[node]["relative_coord"]) + [G.nodes[node]["signal_strength"]])
        
        node_attrs = np.array(node_attrs)
        # print(node_attrs)
        # get edge attrs
        edge_attrs = []
        for edge in G.edges:
            edge_attrs.append(G.edges[edge]["distance"])

        edge_attrs = np.array(edge_attrs)
        # get labels from scan coordinates
        labels = torch.tensor(np.array(list(relative_scan_coords)), dtype=torch.float)

        # get data object
        data = Data(x=torch.tensor(node_attrs, dtype=torch.float), edge_index=edge_index, edge_attr=torch.tensor(edge_attrs, dtype=torch.float), y=labels)

        return data
