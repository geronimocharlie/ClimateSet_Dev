import glob
import os
import json
import tables
import argparse
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from plot_map import plot_timeserie, plot_average, plot_gif


def convert_netcdf_to_pandas(filename: str, features_name: list, frequency:
                             str, remove_poles: bool):
    """
    Convert a NetCDF file to a pandas dataframe using xarray.
    Args:
        filename: name of a NetCDF file
        features_name: name of feature to consider, if empty consider all
        frequency: frequency to aggregate the data (day|week|month)
    Returns:
        df, coordinates, ds.attrs, features_name: the dataframe, the latitude x
        longitude, the NetCDF metadata and the features_name used
    """
    # open dataset with xarray and convert it to a pandas DataFrame
    # ds = xr.open_dataset(filename)
    ds = xr.load_dataset(filename, engine="cfgrib")

    df = ds.to_dataframe()
    ds.close()
    df = df.reset_index()

    # if no features_name is specified, take all columns that are not latitude,
    # longitude and time
    if not features_name:
        features_name = list(set(df.columns) - {"lat", "lon", "time"})
    columns_to_keep = ["lat", "lon"] + features_name

    if remove_poles:
        df = df[(df["lat"] != -90.) & (df["lat"] != 90.)]

    # average the data over week or month
    # Note: using pd.Grouper can lead to problems for week aggregation as the number
    # of week can vary from 53 to 54 depending on the year.
    if frequency == "day":
        # check if it is a leap year, if so remove 29 Feb
        if df["time"].iloc[0].is_leap_year:
            df = df.drop(df[(df["time"].dt.month == 2) & (df["time"].dt.day == 29)].index)
            print("remove February 29th")
    elif frequency == "week":
        df = df.groupby([pd.Grouper(key='time', freq="W-MON"), "lat", "lon"])[features_name].mean().reset_index()
    elif frequency == "month":
        df = df.groupby([pd.Grouper(key='time', freq="M"), "lat", "lon"])[features_name].mean().reset_index()
    else:
        raise NotImplementedError(f"This value for frequency ({frequency}) is not yet implemented")

    # convert time to timestamp (in seconds)
    df["timestamp"] = pd.to_datetime(df['time']).astype(int) / 10**9

    # keep only lat, lon, timestamp and the feature in 'features_name'
    columns_to_keep = ["timestamp"] + columns_to_keep
    df = df[columns_to_keep]

    # pivot the table so that there is only one row per time
    df_pivoted = df.pivot_table(index="timestamp", columns=["lat", "lon"], values=features_name[0])
    coordinates = np.zeros((df_pivoted.shape[1], 2))
    for i, col in enumerate(df_pivoted.columns):
        coordinates[i, 0] = col[0]
        coordinates[i, 1] = col[1]

    return df_pivoted, coordinates, ds.attrs, features_name


def standardize(df: pd.DataFrame, mean=None, std=None) -> pd.DataFrame:
    """ Remove the mean and divide by the standard deviation """
    if mean is None and std is None:
        mean = df.mean()
        std = df.std()

    df_out = (df - mean) / std
    return df_out


class SeasonRemover:
    """ Remove the seasonal cycle. Basically remove the mean for each day/week of
    the year. Update keeps track of the mean and std in an online fashion """
    def __init__(self, t_per_year, d_x):
        self.count = 0
        self.mean = np.zeros((t_per_year, d_x))
        self.std = np.zeros((t_per_year, d_x))
        self.m2 = np.zeros((t_per_year, d_x))  # this is \sum(x_i - mean) ** 2

    def update(self, data):
        self.count += 1
        delta = data - self.mean
        self.mean += delta / self.count
        delta2 = data - self.mean
        self.m2 += delta * delta2

# def update(self, data):
#     n_a = self.count
#     n_b = data.shape[0]
#
#     __import__('ipdb').set_trace()
#     self.mean = (n_a * self.mean + n_b * data.mean(axis=0)) / (n_a + n_b)
#     self.count += n_b
#
#     m2_a = self.m2
#     m2_b = np.mean((data - data.mean(axis=0)) ** 2)
#     self.m2 = m2_a + m2_b + delta ** 2 * n_a * n_b / (n_a + n_b)


def detrending():
    pass


def find_all_files(directory: str, extension: str = "nc") -> list:
    """
    Find all NetCDF or grib files in 'directory'
    Returns: a list of the files name
    """
    if directory[-1] == "/":
        pattern = f"{directory}*.{extension}"
    else:
        pattern = f"{directory}/*.{extension}"
    filenames = sorted([x for x in glob.glob(pattern)])
    return filenames


def main_numpy(netcdf_directory: str, extension: str,  output_path: str, features_name: list, frequency: str, verbose: bool):
    """
    Convert netCDF4 files from the NCEP-NCAR Reanalysis project to a numpy file.
    All the files are expected to be in the directory `netcdf_directory`
    Returns:
        df, n, coordinates, features_name: the complete dataframe, the number of
        samples, an array of the coordinates and the features_name
    """
    # TODO: could add detrending?
    df = None

    # find all the netCDF4 in the directory `netcdf_directory`
    filenames = find_all_files(netcdf_directory, extension)
    if verbose:
        print(f"NetCDF Files found: {filenames}")

    # convert all netCDF4 files in a directory to a single pandas dataframe
    for filename in filenames:
        if verbose:
            print(f"opening file: {filename}")
        df_temp, coordinates, metadata, features_name = convert_netcdf_to_pandas(filename, features_name, frequency)
        if df is None:
            df = df_temp
        else:
            df = pd.concat([df, df_temp])
        if verbose:
            print(df.shape)

    # convert the dataframe to numpy, create the path if necessary and save it
    data_path = os.path.join(output_path, "data.npy")
    if verbose:
        print(f"All files opened, converting to numpy and saving to {data_path}.")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    df = standardize(df)
    np_array = df.values
    # expand to have axis for n and d, respectively the number of timeseries
    # and of features
    np_array = np.expand_dims(np_array, axis=0)
    np_array = np.expand_dims(np_array, axis=2)
    np.save(data_path, np_array)

    # save a copy of one metadata file
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return df, df.shape[0], coordinates, features_name


def main_hdf5(netcdf_directory: str, extension: str, output_path: str, features_name: list,
              frequency: str, verbose: bool, remove_season: bool = True,
              lat_reweight: bool = False, remove_poles: bool = True):
    """
    Convert netCDF4 files from the NCEP-NCAR Reanalysis project to a hdf5 file.
    Only open one files at a time and append the data to the hdf5 file
    All the files are expected to be in the directory `netcdf_directory`.
    Returns:
        df, n, coordinates, features_name: the first dataframe, the number of
        samples, an array of the coordinates and the features_name of the first file
    """
    df = None
    sections = []
    n = 0

    # find all the netCDF4 in the directory `netcdf_directory`
    filenames = find_all_files(netcdf_directory, extension)
    if verbose:
        print(f"NetCDF Files found: {filenames}")
    __import__('ipdb').set_trace()

    # arr_total = None

    # convert all netCDF4 files in a directory to pandas dataframe and append
    # it to the hdf5 file
    for i, filename in enumerate(filenames):
        if verbose:
            print(f"opening file: {filename}")
        df, coordinates, metadata, features_name = convert_netcdf_to_pandas(filename,
                                                                            features_name,
                                                                            frequency,
                                                                            remove_poles)

        # convert the dataframe to numpy, create the path if necessary and save it
        data_path = os.path.join(output_path, "data.h5")
        if verbose:
            print(df.shape)
            print(f"Converting to numpy and saving to {data_path}.")
            print(df.values.mean())
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # expand to have axis for n and d, respectively the number of timeseries and of features
        np_array = df.values
        np_array = np.expand_dims(np_array, axis=0)
        np_array = np.expand_dims(np_array, axis=2)
        n += np_array.shape[1]
        sections.append(np_array.shape[1])

        # plot data
        if not remove_season:
            timeserie_path = os.path.join(args.output_path, f"timeserie_{i}.png")
            plot_timeserie(df, coordinates, frequency, features_name[0], timeserie_path)

        if i == 0:
            # keep the mean and std per day. Useful for removing the season effect
            if remove_season:
                season_remover = SeasonRemover(df.shape[0], df.shape[1])
                season_remover.update(df.values)

            first_df = df
            first_features_name = features_name

            # create the file for the first step
            f = tables.open_file(data_path, mode='w')
            atom = tables.Float64Atom()
            array = f.create_earray(f.root, 'data', atom, (np_array.shape[0], 0, np_array.shape[2], np_array.shape[3]))
            array.append(np_array)
            f.close()

            # save a copy of the first metadata file
            metadata_path = os.path.join(output_path, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
        else:
            # append data to the existing hdf5 file
            f = tables.open_file(data_path, mode='a')
            f.root.data.append(np_array)
            f.close()
            if remove_season:
                season_remover.update(df.values)

    # repass through the data to remove the seasonal effect
    if remove_season or lat_reweight:
        idx = 0
        f = tables.open_file(data_path, mode='r+')
        for section in sections:
            data = f.root.data[:, idx:idx + section]
            if remove_season:
                m2 = season_remover.m2.reshape(data.shape)
                mean = season_remover.mean.reshape(data.shape)

                std = np.sqrt(m2 / season_remover.count)
                f.root.data[:, idx:idx + section] = (data - mean) / std
            if lat_reweight:
                lat_radian = coordinates[:, 0] * np.pi / 180
                f.root.data[:, idx:idx + section] = data * np.cos(lat_radian)

            idx += section
        f.close()

    # quick reading test
    f = tables.open_file(data_path, mode='r')
    data = f.root.data[0, 5:10]
    if verbose:
        print("Test the hdf5 file, here are the row 5 to 10:")
        print(data)
    f.close()

    return first_df, n, coordinates, first_features_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NetCDF files to numpy or hdf5. Can also plot visualizations.")
    parser.add_argument("--data-path", type=str, default="data/sea_level_pressure",
                        help="Path to the directory containing the NetCDF files")
    parser.add_argument("--extension", type=str, default="nc",
                        help="Extensions of the files (nc or grib)")
    parser.add_argument("--output-path", type=str, default="sea_level_results",
                        help="Path where to save the results.")
    parser.add_argument("--features-name", nargs="+",
                        help="Name of the feature to use, if not specified use all")
    parser.add_argument("--frequency", type=str, default="week",
                        help="Frequency to which we parse the data (day|week|month)")
    parser.add_argument("--verbose", action="store_true",
                        help="If True, print useful messages")
    parser.add_argument("--hdf5", action="store_true",
                        help="If True, save result as an hdf5 file")
    parser.add_argument("--gif-max-step", type=int, default=20,
                        help="Maximal number of step to consider to generate the gif")
    args = parser.parse_args()

    # example of default command:
    # python convert_netcdf_files.py --verbose --frequency week --hdf5

    # TODO: if necessary, adapt to multiple feature. Now, probably only works
    # for one feature

    if args.hdf5:
        df, n, coordinates, features_name = main_hdf5(args.data_path,
                                                      args.extension,
                                                      args.output_path,
                                                      args.features_name,
                                                      args.frequency,
                                                      args.verbose)
    else:
        df, n, coordinates, features_name = main_numpy(args.data_path,
                                                       args.extension,
                                                       args.output_path,
                                                       args.features_name,
                                                       args.frequency,
                                                       args.verbose)

    # save a json containing some parameters of the dataset
    params = {"n": 1,
              "t": n,
              "d": 1,
              "d_x": df.shape[1]}
    json_path = os.path.join(args.output_path, "data_params.json")
    with open(json_path, "w") as file:
        json.dump(params, file, indent=4)

    # save the coordinates
    coord_path = os.path.join(args.output_path, "coordinates.npy")
    np.save(coord_path, coordinates)

    # plot data
    timeserie_path = os.path.join(args.output_path, "timeserie.png")
    average_path = os.path.join(args.output_path, "average.png")
    plot_timeserie(df, coordinates, args.frequency, features_name[0], timeserie_path)
    plot_average(df, coordinates, average_path)
    plot_gif(df, coordinates, args.output_path, args.gif_max_step)