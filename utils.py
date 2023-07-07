"""
Utils for the NY CitiBike Data Science Challenge.
"""

# data science
import numpy as np

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def crow_distance(start_station_latitude,
                  start_station_longitude,
                  end_station_latitude,
                  end_station_longitude):
    """
    Lon, lat in degrees, can be arrays
    output: the distance between the points in meters
    """

    r = 6371000

    lat_1 = start_station_latitude * np.pi / 180.
    lon_1 = start_station_longitude * np.pi / 180.
    lat_2 = end_station_latitude * np.pi / 180.
    lon_2 = end_station_longitude * np.pi / 180.

    x_1 = r * np.cos(lat_1) * np.cos(lon_1)
    y_1 = r * np.cos(lat_1) * np.sin(lon_1)
    z_1 = r * np.sin(lat_1)
    x_2 = r * np.cos(lat_2) * np.cos(lon_2)
    y_2 = r * np.cos(lat_2) * np.sin(lon_2)
    z_2 = r * np.sin(lat_2)

    return np.sqrt((x_1-x_2)**2 + (z_1-z_2)**2 + (z_1-z_2)**2)


def forward_feature_selection(data, feature_columns, target_column):
    """
    Choose features for the random forest
    """

    # create pairs
    pair_list = []
    for f1 in feature_columns:
        for f2 in feature_columns:
            condition1 = f1 != f2
            condition2 = [f2, f1] not in pair_list
            if (condition1 and condition2):
                pair_list.append([f1, f2])

    # train on all pairs and append to acc list
    print('\ntesting 2 features:')
    acc_list = []
    for pair in pair_list:
        x_train = data.loc[data.train_set, pair]
        x_dev = data.loc[data.dev_set, pair]
        y_train = data.loc[data.train_set, target_column]
        y_dev = data.loc[data.dev_set, target_column]
        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(x_train, y_train.to_numpy().reshape(-1))
        y_dev_hat = rf.predict(x_dev)
        acc = accuracy_score(y_dev, y_dev_hat)
        acc_list.append(acc)
        print(f"{', '.join(pair)}: acc = {acc:.4f}")

    # look for the winning feature
    chosen_features = pair_list[acc_list.index(max(acc_list))]
    best_acc = max(acc_list)
    print('\nwinners:')
    print(f"{', '.join(chosen_features)}: acc = {best_acc:.4f}")

    # append new features iteratively until acc decreases
    for i in range(len(feature_columns)-2):
        chosen_features_old = chosen_features
        best_acc_old = best_acc
        print(f'\ntesting {len(chosen_features)+1} features:')
        candidates_list = []
        new_acc_list = []
        for feature in feature_columns:
            if feature in chosen_features_old:
                continue
            candidates = chosen_features + [feature]
            x_train = data.loc[data.train_set, candidates]
            x_dev = data.loc[data.dev_set, candidates]
            rf.fit(x_train, y_train.to_numpy().reshape(-1))
            y_dev_hat = rf.predict(x_dev)
            acc = accuracy_score(y_dev, y_dev_hat)
            new_acc_list.append(acc)
            candidates_list.append(candidates)
            print(f"{', '.join(candidates)}: acc = {acc:.4f}")
        chosen_features = candidates_list[new_acc_list.index(
                                             max(new_acc_list))]
        best_acc = max(new_acc_list)
        print('winners:')
        print(f"{', '.join(chosen_features)}, acc = {best_acc:.4f}")
        if best_acc < best_acc_old:
            print('break loop, adding these features harmed generalizability')
            # final result
            print('\nfinal result')
            print(f"{', '.join(chosen_features_old)}, acc = {best_acc_old:.4f}")
            break



if __name__ == '__main__':
    """
    Simple tests
    """

    lat_1 = 40.767272
    lon_1 = -73.993929
    lat_2 = 40.760683
    lon_2 = -73.984527

    print('crow_distance_m from a calculator = 1078.8')
    print(f'from our helper function: {crow_distance(lat_1, lon_1, lat_2, lon_2)}')

