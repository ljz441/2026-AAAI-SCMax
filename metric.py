import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
    return acc

def purity(y_true, y_pred):
    y_true = y_true.astype(np.int64).copy()
    y_pred = y_pred.astype(np.int64)
    y_voted_labels = np.zeros_like(y_true)

    labels = np.unique(y_true)
    label_mapping = {label: idx for idx, label in enumerate(labels)}
    y_true = np.array([label_mapping[label] for label in y_true])

    bins = np.arange(len(labels) + 1)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def f1_with_hungarian(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = np.zeros(D, dtype=np.int64)
    mapping[row_ind] = col_ind
    y_pred_aligned = mapping[y_pred]

    return f1_score(y_true, y_pred_aligned, average='macro')

def evaluate(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = cluster_acc(y_true, y_pred)
    pur = purity(y_true, y_pred)
    f1 = f1_with_hungarian(y_true, y_pred)

    return nmi, ari, acc, pur, f1
