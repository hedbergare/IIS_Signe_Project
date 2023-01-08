import numpy as np
import os


def get_GR_data():
    # Read data from files
    path_of_dir = os.getcwd()
    print("DIR PATH", path_of_dir)
    X_path = os.path.join(path_of_dir, path_of_dir, "Data\\all_X.npy")
    Y_path = os.path.join(path_of_dir, path_of_dir, "Data\\all_Y.npy")
    X_data = np.load(X_path, allow_pickle=True)
    Y_data = np.load(Y_path, allow_pickle=True)

    # Make them numpy arrays
    X_array = np.array(X_data)
    Y_array = np.array(Y_data)

    print("Mean before", np.mean(X_array))
    X_array = normalize_by_joint_distance(X_array)
    print("Mean after", np.mean(X_array))

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
