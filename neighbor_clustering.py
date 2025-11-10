import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import torch

class NeighborClustering:
    def __init__(self):
        pass

    def step(self, Z: torch.Tensor, labels: np.ndarray):
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)

        if num_clusters <= 1:
            return num_clusters, labels

        centers = []
        for lbl in unique_labels:
            cluster_points = Z[labels == lbl]
            center = torch.mean(cluster_points, dim=0)
            centers.append(center)
        centers = torch.stack(centers)

        centers_np = centers.cpu().numpy()
        tree = cKDTree(centers_np)
        distances, indices = tree.query(centers_np, k=2)
        neighbor_dists = distances[:, 1]
        neighbor_indices = indices[:, 1]

        num_pairs_to_connect = max(1, int(num_clusters))
        sorted_indices = np.argsort(neighbor_dists)
        selected_indices = sorted_indices[:num_pairs_to_connect]

        adj_matrix = np.zeros((num_clusters, num_clusters), dtype=float)
        for idx in selected_indices:
            i = idx
            j = neighbor_indices[idx]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

        G_sparse = coo_matrix(adj_matrix)
        n_components, cluster_labels = connected_components(G_sparse, directed=False, return_labels=True)

        label_mapping = {old: new for old, new in zip(unique_labels, cluster_labels)}
        new_labels = np.array([label_mapping[lbl] for lbl in labels])

        return n_components, new_labels
