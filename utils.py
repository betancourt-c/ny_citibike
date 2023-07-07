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


def forward_feature_selection(dataset, training_columns, target_column):
    """
    Choose features for the random forest
    """

    # create pairs
    pair_list = []
    for f1 in training_columns:
        for f2 in all_features:
            condition1 = f1 != f2
            condition2 = [f2, f1] not in pair_list
            if (condition1 and condition2):
                pair_list.append([f1, f2])

    # train on all pairs and append to r2 list
    print('\ntesting 2 features:')
    acc_list = []
    for pair in pair_list:
        x_train = None
        y_train = None
        x_val = None
        y_val = None
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        y_val_hat = rf.predict(x_val)
        acc = accuracy_score(y_val, y_val_hat)
        acc_list.append(acc)
        print(f"{', '.join(pair)}: acc = {acc:.4f}")

    # look for the winning feature
    chosen_features = pair_list[acc_list.index(max(acc_list))]
    best_acc = max(acc_list)
    print('\nwinners:')
    print(f"{', '.join(chosen_features)}: acc = {best_acc:.4f}")

    # append new features iteratively until r2 decreases
    for i in range(len(all_features)-2):
        chosen_features_old = chosen_features
        best_acc_old = best_acc
        print(f'\ntesting {len(chosen_features)+1} features:')
        candidates_list = []
        new_acc_list = []
        for feature in all_features:
            if feature in chosen_features_old:
                continue
            candidates = chosen_features + [feature]
            x_train = None
            x_val = None
            rf = RandomForestClassifier()

            y_hat_val = rf.y_hat.numpy().reshape(-1)[mask_df['val_mask']]
            r2 = r2_score(y_val, y_hat_val)
            candidates_list.append(candidates)
            new_r2_list.append(r2)
            print(f"{', '.join(candidates)}: R2 = {r2:.4f}")
        chosen_features = candidates_list[new_r2_list.index(
                                             max(new_r2_list))]
        best_r2 = max(new_r2_list)
        print('winners:')
        print(f"{', '.join(chosen_features)}, R2 = {best_r2:.4f}")
        if best_r2 < best_r2_old:
            print('break loop, adding these features harmed generalizability')
            # final result
            print('\nfinal result')
            print(f"{', '.join(chosen_features_old)}, R2 = {best_r2_old:.4f}")
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

