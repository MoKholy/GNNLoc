import pandas as pd
import numpy as np


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
    return coordinates["x"].to_numpy(), coordinates["y"].to_numpy(), coordinates["z"].to_numpy()

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