import numpy as np
from scipy.spatial.distance import cdist

def contextual_distance(query_embeddings: np.ndarray, video_embeddings: np.ndarray, metric: str = 'sqeuclidean') -> float:
    """
    Calculates the contextual distance between query and video embeddings.

    Args:
        query_embeddings: Embeddings of the generated frames (k, d).
        video_embeddings: Embeddings of the video frames (T, d).
        metric: The distance metric to use.

    Returns:
        The contextual distance.
    """
    # Compute pairwise distances between all query and video frames
    pairwise_dist = cdist(query_embeddings, video_embeddings, metric=metric)

    # For each query frame, find the minimum distance to any video frame
    min_distances = np.min(pairwise_dist, axis=1)

    # The contextual distance is the sum of these minimum distances
    return np.sum(min_distances)


def velocity_aware_dtw(query_embeddings: np.ndarray, video_embeddings: np.ndarray, beta: float = 1.0, metric: str = 'sqeuclidean') -> float:
    """
    Calculates the action distance using Velocity-Aware Dynamic Time Warping.

    Args:
        query_embeddings: Embeddings of the generated frames (k, d).
        video_embeddings: Embeddings of the video frames (T, d).
        beta: Weight for the velocity component.
        metric: The distance metric to use for static appearance.

    Returns:
        The action distance.
    """
    k, d = query_embeddings.shape
    T, _ = video_embeddings.shape

    if T < k:
        return float('inf')

    # Compute velocity curves
    delta_q = np.diff(query_embeddings, axis=0)
    delta_c = np.diff(video_embeddings, axis=0)

    # Initialize DP table
    D = np.full((k, T), float('inf'))

    # Local cost function
    def local_cost(j, t):
        dist = cdist(query_embeddings[j:j+1], video_embeddings[t:t+1], metric=metric)[0, 0]
        if j < k - 1 and t < T - 1:
            vel_dist = cdist(delta_q[j:j+1], delta_c[t:t+1], metric=metric)[0, 0]
            dist += beta * vel_dist
        return dist

    # Fill DP table
    for t in range(T):
        D[0, t] = local_cost(0, t)

    for j in range(1, k):
        min_prev_cost = float('inf')
        for t in range(1, T):
            min_prev_cost = min(min_prev_cost, D[j-1, t-1])
            D[j, t] = local_cost(j, t) + min_prev_cost

    # Action distance is the square root of the minimum cost in the last row
    min_cost = np.min(D[k-1, :])
    return np.sqrt(min_cost) if min_cost != float('inf') else float('inf')
