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
    X_path = os.path.join(path_of_dir, path_of_dir, "GR\\Data\\new_X.npy")
    Y_path = os.path.join(path_of_dir, path_of_dir, "GR\\Data\\new_Y.npy")
    X_data = np.load(X_path, allow_pickle=True)
    Y_data = np.load(Y_path, allow_pickle=True)

    # Make them numpy arrays
    X_array = np.array(X_data)
    Y_array = np.array(Y_data)

    # Split into training and test sets
    divisor = round(0.8*len(X_array))
    for i in range(divisor, divisor+20):
        if (i % 20 == 0):
            train_size = i
            break

    assert len(Y_array) == len(X_array)

    X_train = X_array[0:train_size]
    X_test = X_array[train_size:]

    Y_train = Y_array[0:train_size]
    Y_test = Y_array[train_size:]

    return X_train, X_test, Y_train, Y_test


def normalize_by_min_max(X_values):
    for frame in X_values:
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    return X_values


def normalize_by_joint_distance(X_values):
    for i in range(len(X_values)):
        distance = 0
        # At least one of the following joint distances exists for all frames!
        if (X_values[i][6][7] != 0):
            distance = X_values[i][6][7]

        elif (X_values[i][10][11] != 0):
            distance = X_values[i][10][11]

        elif (X_values[i][14][15] != 0):
            distance = X_values[i][14][15]

        elif (X_values[i][18][19] != 0):
            distance = X_values[i][18][19]

        elif (X_values[i][2][3] != 0):
            distance = X_values[i][2][3]

        X_values[i] = X_values[i] / distance
    return X_values


X_train, X_test, Y_train, Y_test = get_GR_data()
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print("Mean:", np.mean(X_train))
print("Max:", np.max(X_train))
print("Min:", np.min(X_train))
#X_train = normalize_data(X_train, Y_train)
X_train = normalize_by_joint_distance(X_train)
print("Mean after:", np.mean(X_train))
print("Max after:", np.max(X_train))
print("Min after:", np.min(X_train))
# print(np.mean(X_train))
