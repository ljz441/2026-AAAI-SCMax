import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize


def load_data(data_name, is_normalize=True):
    X_list = []
    if data_name in ['MSRCv1', 'HW2', 'Wiki', 'MNIST', 'YTF20']:
        data_dir = './data/' + data_name + '.mat'
        mat = sio.loadmat(data_dir)
        Y = np.squeeze(mat['Y'])
        X = mat['X'].reshape(-1)
        view_size = X.shape[0]
    elif data_name in ['Cifar10', 'Cifar100', 'Fashion']:
        data_dir = './data/' + data_name + '.mat'
        mat = sio.loadmat(data_dir)
        Y = np.squeeze(mat['Y'][0,0])
        X = mat['X'].reshape(-1)
        view_size = X.shape[0]
    else:
        raise Exception('Wrong data name!')

    if is_normalize:
        if Y.shape[0] == X[0].shape[0]:
            for view in range(view_size):
                X_list.append(normalize(X[view], norm='l2'))
        else:
            for view in range(view_size):
                X_list.append(normalize(X[view].T, norm='l2'))
    else:
        if Y.shape[0] == X[0].shape[0]:
            for view in range(view_size):
                X_list.append(X[view])
        else:
            for view in range(view_size):
                X_list.append(X[view].T)

    Y = Y.astype(np.int32)
    dims = [x.shape[1] for x in X_list]

    return X_list, Y, dims