import torch
import argparse
import random
import time
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from auto_encoder import AutoEncoder
from data_loader import load_data
from neighbor_clustering import NeighborClustering
from feature_optimization import BackFeature
from metric import evaluate

Dataname = 'HW2'
parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--back_feature_epochs", default=50)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--seed", default=3407)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def pretrain_all():
    total_steps = args.mse_epochs * data_size
    global_step = 0
    tot_loss = 0.
    epoch_losses = []

    with tqdm(total=total_steps, ncols=120) as pbar:
        for epoch in range(1, args.mse_epochs + 1):
            auto_encoder.train()

            indices = list(range(data_size))
            np.random.shuffle(indices)

            for start in range(0, data_size, args.batch_size):
                end = min(start + args.batch_size, data_size)
                batch_idx = indices[start:end]

                xs = []
                for v in range(view_size):
                    x_view = torch.tensor(X_list[v][batch_idx], dtype=torch.float32).to(device)
                    xs.append(x_view)

                optimizer.zero_grad()
                xrs, zs = auto_encoder(xs)
                loss_list = [criterion(xrs[v], xs[v]) for v in range(view_size)]
                loss = sum(loss_list)

                loss.backward()
                optimizer.step()

                batch_size = end - start
                tot_loss += loss.item() * batch_size
                global_step += batch_size

                avg_loss = tot_loss / global_step
                epoch_losses.append(avg_loss)
                pbar.set_description(f"Epoch {epoch}/{args.mse_epochs}")
                pbar.set_postfix({'MSE Loss': f'{avg_loss:.6f}'})
                pbar.update(batch_size)

        # file_name = f'epoch_losses_round_AE.xlsx'
        # df = pd.DataFrame({'Autoencoder': epoch_losses})
        # df.to_excel(file_name, index=False)

def train_back_feature_all(labels, Ki):
    total_steps = args.back_feature_epochs * data_size
    global_step = 0
    tot_loss = 0.
    epoch_losses = []

    back_feature_module = BackFeature(args.batch_size, Ki, device)

    with tqdm(total=total_steps, ncols=120) as pbar:
        for epoch in range(1, args.back_feature_epochs + 1):
            auto_encoder.train()

            indices = list(range(data_size))
            np.random.shuffle(indices)

            for start in range(0, data_size, args.batch_size):
                end = min(start + args.batch_size, data_size)
                batch_idx = indices[start:end]

                xs = []
                for v in range(view_size):
                    x_view = torch.tensor(X_list[v][batch_idx], dtype=torch.float32).to(device)
                    xs.append(x_view)

                batch_labels = labels[batch_idx]

                optimizer.zero_grad()
                xrs, zs = auto_encoder(xs)

                loss_list = []
                for v in range(view_size):
                    loss_list.append(back_feature_module.forward_label(zs[v], batch_labels))
                    loss_list.append(criterion(xrs[v], xs[v]))
                loss = sum(loss_list)

                loss.backward()
                optimizer.step()

                batch_size = end - start
                tot_loss += loss.item() * batch_size
                global_step += batch_size

                avg_loss = tot_loss / global_step
                epoch_losses.append(avg_loss)
                pbar.set_description(f"Epoch {epoch}/{args.back_feature_epochs}")
                pbar.set_postfix({'Back Loss': f'{avg_loss:.6f}'})
                pbar.update(batch_size)

    # file_name = f'epoch_losses_round_{Ki}.xlsx'
    # df = pd.DataFrame({'Autoencoder': epoch_losses})
    # df.to_excel(file_name, index=False)

def best_mapping(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    D = max(y_pred.max(), y_true.max()) + 1
    confusion = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        confusion[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    total_correct = confusion[row_ind, col_ind].sum()
    reward = total_correct / len(y_pred)
    return reward

setup_seed(args.seed)

print('\n*** Load Data ***\n')
X_list_all, Y, dims_all = load_data(args.dataset, is_normalize=True)
view_size_all = len(X_list_all)
data_size = Y.shape[0]
k = np.unique(Y).shape[0]
print(f'Data Name: {args.dataset}, View Number: {view_size_all}, Data Size: {data_size}, Cluster Number: {k}')

start_time = time.time()
X_list = [X_list_all[0]]
view_size = len(X_list)
dims = [dims_all[0]]

print('\n*** Auto Encoder ***\n')
auto_encoder = AutoEncoder(view_size, dims, args.feature_dim, device)
optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=args.learning_rate)
criterion = torch.nn.MSELoss()
pretrain_all()

print('\n*** Cluster Number Estimation ***\n')
X_tensor = [torch.tensor(x, dtype=torch.float32).to(device) for x in X_list]
auto_encoder.eval()
with torch.no_grad():
    Z_concat = auto_encoder.forward_all_z(X_tensor)
clusterer = NeighborClustering()
labels = np.arange(data_size)

Ki_list = []
Ki_scores = {}
Z_pre = None
labels_pre = None
best_score = -1
best_Ki = None
best_labels = None
best_labels_pre = None

while True:
    Ki, labels = clusterer.step(Z_concat, labels)
    Ki_list.append(Ki)
    print(f'Current Ki: {Ki}')
    if Z_pre is not None:
        K_pre, labels_pre = clusterer.step(Z_pre, labels_pre)
        print(f'Current Ki_new: {K_pre}')
        if Ki < 3 or K_pre < 3: break   # K ≥ 3

        score = best_mapping(labels, labels_pre)

        if score > best_score:
            best_score = score
            best_Ki = Ki
            best_labels = labels.copy()
            best_labels_pre = labels_pre.copy()

        Ki_scores[Ki] = score
        print(f"Ki = {Ki}, Score = {score:.4f}")
    labels_pre = labels

    time.sleep(0.05)
    train_back_feature_all(labels, Ki)
    auto_encoder.eval()
    with torch.no_grad():
        Z_pre = auto_encoder.forward_all_z(X_tensor)

print("Ki list:", Ki_list)
print("\nAll Ki Scores:")
for Ki in Ki_scores:
    print(f"Ki = {Ki}, Score = {Ki_scores[Ki]:.4f}")
print(f"\nBest Ki: {best_Ki}, Highest Score: {best_score:.4f}")

if best_labels is not None and len(best_labels) > 0:
    nmi, ari, acc, pur, f1 = evaluate(Y, best_labels)
    print(f'NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, Purity: {pur:.4f}, F1: {f1:.4f}')
else:
    print("best_labels is empty or None, final K = 1.")
    best_labels = np.zeros(data_size, dtype=int)
    nmi, ari, acc, pur, f1 = evaluate(Y, best_labels)
    print(f'NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, Purity: {pur:.4f}, F1: {f1:.4f}')

if best_labels_pre is not None and len(best_labels_pre) > 0:
    nmi, ari, acc, pur, f1 = evaluate(Y, best_labels_pre)
    print(f'NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, Purity: {pur:.4f}, F1: {f1:.4f}')
else:
    print("best_labels_pre is empty or None, final K = 1.")
    best_labels_pre = np.zeros(data_size, dtype=int)
    nmi, ari, acc, pur, f1 = evaluate(Y, best_labels_pre)
    print(f'NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, Purity: {pur:.4f}, F1: {f1:.4f}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total Time: {elapsed_time:.6f} seconds")