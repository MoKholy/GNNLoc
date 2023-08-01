import pandas as pd
import numpy as np
import networkx as nx


# function to collect indices of aps heard, and their rssi values
def get_location_indices_signals(df):
    # empty list for all ap points heard and all ap signal strengths
    all_ap_points, all_ap_signals = [], []
    for index, colname in enumerate(df):
        # ignore 100 db readings as they are filler as readMe file said
        # print("Done with AP {}".format(index-1))
        idx = df[colname].index[df[colname] != 100].to_list()
        all_ap_points.append(idx)
        all_ap_signals.append(df[colname].loc[idx])
    return all_ap_points, all_ap_signals

# function to get coordinates of scans
def get_index_coordinates(indices, coords_df):
    # get the coordinates of the aps using indices
    coordinates = coords_df.loc[indices]
    return coordinates.iloc[0].to_numpy(), coordinates.iloc[1].to_numpy() #, coordinates.iloc[2].to_numpy()



# function to estimate ap weight
def get_ap_weight(rss):
    return 100**(rss/10.0)

# function to estimate ap coordinates
def approximate_ap_coordinates(x_coords, y_coords, z_coords, weights):

    # sum product of weight and coordinate
    estimated_x = np.sum(x_coords * weights)
    estimated_y = np.sum(y_coords * weights)
    estimated_z = np.sum(z_coords * weights)

    # get denominator
    sum_of_weights = weights.sum()
    
    # return tuple of coordinates
    return estimated_x/sum_of_weights , estimated_y/sum_of_weights, estimated_z/sum_of_weights

# function to estimate all ap coordinates
def approximate_all_ap_locations(data, coords):
    # get location and signals
    all_ap_points, all_ap_signals = get_location_indices_signals(data)
    estimated_ap_locs = []
    # check the length of them both are the same
    assert len(all_ap_points) == len(all_ap_signals)
    for i in range(len(all_ap_points)):
        # get ap indices and rss strengths
        idx, signals = all_ap_points[i], all_ap_signals[i]
        # get coordinates of the scans
        xs, ys, zs = get_index_coordinates(idx, coords)
        # get weights
        weights = np.asarray([get_ap_weight(signal) for signal in signals])
        # get estimated coordinates
        estimated_coords = approximate_ap_coordinates(xs, ys, zs, weights)
        estimated_ap_locs.append(list(estimated_coords))
    return np.asarray(estimated_ap_locs)


# def funciton to save the data and return the dataframe
def save_estimated_coordinates(scans, coords, save_loc):
    # get the estimated coordinates
    estimated_coords = approximate_all_ap_locations(scans, coords)
    # create dataframe
    df = pd.DataFrame(estimated_coords, columns=["x", "y", "z"])
    # save the dataframe
    df.to_csv(save_loc, index=False)
    return df

# function to find index of strongest AP index
def get_strongest_ap_index(scan):
    # make copy of scan
    scan_copy = np.copy(scan)
    # set all 100s to -1000
    scan_copy[scan_copy == 100] = -1000
    # return index of strongest ap
    return np.argmax(scan_copy)

# function to get heard AP indices and signal strengths
def get_heard_ap_indices_and_signals(scan):
    ap_heard_indices = np.argwhere(scan !=100)
    ap_heard_signals = scan[ap_heard_indices]
    return ap_heard_indices, ap_heard_signals

# function to get the top k heard AP indices and signal strengths per scan
def get_top_k_heard_ap_indices_and_signals(scan, k=10):
    # get the heard ap indices and signals
    ap_heard_indices, ap_heard_signals = get_heard_ap_indices_and_signals(scan)
    # sort the signals
    assert len(ap_heard_indices) == len(ap_heard_signals)

    # flatten the arrays
    ap_heard_indices = ap_heard_indices.flatten()
    ap_heard_signals = ap_heard_signals.flatten()
    sorted_indices = np.argsort(ap_heard_signals)
    # print("sortred indices: {}".format(sorted_indices))
    # print("heard indices: {}".format(ap_heard_indices))
    # print("heard signals: {}".format(ap_heard_signals))
    # get the top k indices and signals
    if len(ap_heard_indices > k):
        top_k_indices = ap_heard_indices[sorted_indices[-k:]]
        top_k_signals = ap_heard_signals[sorted_indices[-k:]]
    else:
        top_k_indices = ap_heard_indices
        top_k_signals = ap_heard_signals
    
    # return the top k indices and signals

    return top_k_indices, top_k_signals

# function to get the relative coordinates wrt to the strongest ap
def get_relative_coordinates(scan, coords, scan_coords, k="none"):

    # get the index of the strongest ap
    strongest_ap_index = get_strongest_ap_index(scan)

    # get the coordinates of the strongest ap
    strongest_ap_coords = coords.loc[strongest_ap_index]

    # get the heard ap indices and signals

    # if k is none, get all the heard ap indices and signals, else get the k strongest ap indices and signals
    if k == "none":
        ap_heard_indices, ap_heard_signals = get_heard_ap_indices_and_signals(scan)
    else:
        ap_heard_indices, ap_heard_signals = get_top_k_heard_ap_indices_and_signals(scan, k=k)
        
    # get the coordinates of the heard aps
    ap_heard_coords = coords.loc[ap_heard_indices.flatten()].to_numpy()

    # get the relative coordinates
    relative_coords = []
    for ap_heard_coord in ap_heard_coords:
        # subtract the strongest ap coordinates from the heard ap coordinates
        # ap_x, ap_y, ap_z = ap_heard_coord
        ap_x, ap_y = ap_heard_coord
        # strongest_ap_x, strongest_ap_y, strongest_ap_z = strongest_ap_coords
        strongest_ap_x, strongest_ap_y = strongest_ap_coords
        relative_coords.append((ap_x - strongest_ap_x, ap_y - strongest_ap_y))
        # relative_coords.append((ap_x - strongest_ap_x, ap_y - strongest_ap_y, ap_z - strongest_ap_z))

    # get relative coordinates of the scan
    # scan_x, scan_y, scan_z = scan_coords
    scan_x, scan_y = scan_coords
    # strongest_ap_x, strongest_ap_y, strongest_ap_z = strongest_ap_coords
    strongest_ap_x, strongest_ap_y = strongest_ap_coords
    # relative_scan_coords = (scan_x - strongest_ap_x, scan_y - strongest_ap_y, scan_z - strongest_ap_z)
    relative_scan_coords = (scan_x - strongest_ap_x, scan_y - strongest_ap_y)
    # return the relative coordinates
    return relative_coords, ap_heard_signals, relative_scan_coords, strongest_ap_coords

def get_distance_between_aps(ap1, ap2, distance_metric="euclidean"):
    # get the coordinates of the aps
    # x1, y1, z1 = ap1
    # x2, y2, z2 = ap2
    x1, y1 = ap1
    x2, y2 = ap2
    # calculate the distance based on distance metric
    # distance can be euclidean, manhattan, or inverse of euclidean, default to euclidean
    if distance_metric == "euclidean":
        # distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    elif distance_metric == "manhattan":
        # distance = np.abs(x1-x2) + np.abs(y1-y2) + np.abs(z1-z2)
        distance = np.abs(x1-x2) + np.abs(y1-y2)
    elif distance_metric == "inverse_euclidean":
        # distance = 1/(np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)+1)
        distance = 1/(np.sqrt((x1-x2)**2 + (y1-y2)**2)+1)
    else:
        # distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)

    return distance

# function to get distance between all APs
def get_distance_between_all_aps(ap_coords, distance_metric="euclidean"):

    # get the distance between each pair of aps as tuple of (ap1, ap2, distance)
    distances = []
    for i in range(len(ap_coords)):
        for j in range(i+1, len(ap_coords)):
            distances.append((i, j, get_distance_between_aps(ap_coords[i], ap_coords[j], distance_metric)))
    
    return distances

# function to create a graph from the relative distance for a scan, nodes have their relative coordinates and signal strength as features
def create_graph_from_scan(ap_coords, scan, scan_coords, distance_metric="euclidean", k="none"):
    # create a graph
    G = nx.Graph()
    # get the relative coordinates and signal strengths
    relative_coords, ap_heard_signals, normalized_scan_coords, strongest_ap_coordinates = get_relative_coordinates(scan, ap_coords, scan_coords, k=k)
    # add nodes to the graph
    for i in range(len(relative_coords)):
        # get the relative coordinates and signal strength of the node
        relative_coord = relative_coords[i]
        signal_strength = ap_heard_signals[i]
        # add the node to the graph
        G.add_node(i, relative_coord=relative_coord, signal_strength=signal_strength)
    # get the distances between all the aps
    distances = get_distance_between_all_aps(relative_coords, distance_metric)
    # add the edges to the graph
    for distance in distances:
        # get the ap indices and distance
        ap1, ap2, distance = distance
        # add the edge to the graph
        G.add_edge(ap1, ap2, distance=distance)
    # return the graph
    return G, normalized_scan_coords, strongest_ap_coordinates

# function visualize the graph
def visualize_graph(G):
    from matplotlib import pyplot as plt
    # get the positions of the nodes
    pos = nx.spring_layout(G)
    # draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=100)
    # draw the edges
    nx.draw_networkx_edges(G, pos)
    # draw the labels
    nx.draw_networkx_labels(G, pos)
    # show the plot
    plt.show()