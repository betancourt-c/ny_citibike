"""
Utils for the NY CitiBike Data Science Challenge.
"""

# data science
import numpy as np

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

