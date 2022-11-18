import os
import zipfile
import numpy as np
import torch
import csv
def get_adjacency_matrix1(distance_df_filename, num_of_vertices):

    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1]), float(i[2])) for i in reader]
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)
    B = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)
    for i, j, k in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A
# processing PEMS04 data
def load_metr_la_data4():

    distance_df_filename = '../data/PEMS04/distance.csv'
    num_of_vertices = 307
    A = get_adjacency_matrix1(distance_df_filename, num_of_vertices)
    X = np.load("data/PEMS04/pems04.npz")
    F = X['data'][:, :, 0:1]
    S = X['data'][:, :, 2:3]
    X = np.concatenate((F, S), 2)  # splicing traffic flow and speed
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)  # computational efficiency while maintaining data accuracy
    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))  # means为(2,)，means.reshape(1, -1, 1).shape为(1, 2, 1)
    stds = np.std(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    X = X / stds.reshape(1, -1, 1)
    return A, X, means, stds


# processing PEMS08 data
def load_metr_la_data8():

    distance_df_filename = '../data/PEMS08/distance.csv'
    num_of_vertices = 170
    A = get_adjacency_matrix1(distance_df_filename, num_of_vertices)
    X = np.load("data/PEMS08/pems08.npz")
    F = X['data'][:, :, 0:1]
    S = X['data'][:, :, 2:3]
    X = np.concatenate((F, S), 2)  # splicing traffic flow and speed
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32) #
    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    stds = np.std(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    X = X / stds.reshape(1, -1, 1)
    return A, X, means, stds

# 08 over


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_features, num_timesteps_input, num_vertices).
        - Node targets for the samples. Shape is
          (num_vertices, num_features, num_samples).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]
    # Save samples
    # print(indices)   (0, 22), (1, 23), (2, 24), (3, 25), (4, 26),
    # print('X', X.shape)  (307, 2, 10195) (307, 2, 3398) (307, 2, 3399)
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (1, 2, 0)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))
