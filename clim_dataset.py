import os
import datetime
import numpy as np
import xarray as xr

class Climdata(object):



    def __init__(self, data_path):
        self.data_path = data_path

    def read_dataset(self, parameter):
        """
        Reads netCDF dataset using xarray. Currently file names
        are hardcoded! TODO
        Args:
            parameter - (int) grib_id of the dataset to read.
        Returns
            xarray dataframe
        """
        file_name = str(parameter) + '_0.25deg_tmp.nc'
        dataset_path = os.path.join(self.data_path, file_name)
        dataset = xr.open_dataset(dataset_path)
        return dataset

    def spatial_subset(self, dataset, bbox):
        """
        Selects data within spatial bbox. bbox coords must be given as
        positive values for the Northern hemisphere, and negative for 
        Southern. West and East both positive - Note - the method is 
        naive and will only work for bboxes fully fitting in the Eastern hemisphere!!!
        Args:
            dataset - xarray dataset
            bbox - (list) [North, South, West, East]
        Returns:
            xarray dataset
        """
        dataset = dataset.where((dataset.latitude < bbox[0]) & 
                                (dataset.latitude > bbox[1]), drop=True)
        dataset = dataset.where((dataset.longitude > bbox[2]) & 
                                (dataset.longitude < bbox[3]), drop=True)
        return dataset



    def time_subset(self, dataset, hour, start_date=None, end_date=None):
        """
        Selects data within spatial bbox.
        Args:
            dataset - xarray dataset
            hour - (int) hour
        Returns:
            xarray dataset
        """
        dataset = dataset.sel(time=datetime.time(hour))
        return dataset

    def 



if __name__ == '__main__':
    data_path = '/home/tadas/tofewsi/data/'

    #Riau bbox
    bbox = [3, -2, 99, 104]

    ds = Climdata(data_path)

