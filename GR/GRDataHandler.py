import numpy as np
import os


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_GR_data():
    # Read data from files
    path_of_dir = os.getcwd()
    print("DIR PATH", path_of_dir)
    X_path = os.path.join(path_of_dir, path_of_dir, "GR\\Data\\X_train.npy")
    Y_path = os.path.join(path_of_dir, path_of_dir, "GR\\Data\\Y_train.npy")
    X_data = np.load(X_path, allow_pickle=True)
    Y_data = np.load(Y_path, allow_pickle=True)

    X_shuffled, Y_shuffled = unison_shuffled_copies(X_data, Y_data)

    # Make them numpy arrays
    X_array = np.array(X_shuffled)
    Y_array = np.array(Y_shuffled)

    # Normalize X-vectors
    X_array = (X_array - np.min(X_array))/(np.max(X_array) - np.min(X_array))

    # Split into training and test sets
    assert len(Y_array) == len(X_array)
    train_size = round(0.8 * len(X_array))

    X_train = X_array[0:train_size]
    X_test = X_array[train_size:]

    Y_train = Y_array[0:train_size]
    Y_test = Y_array[train_size:]

    return X_train, X_test, Y_train, Y_test


#print("DIRECTORY DATA", os.listdir("./Data/"))


X_train, X_test, Y_train, Y_test = get_GR_data()
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
