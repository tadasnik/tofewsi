import os
import datetime
import numpy as np
import xarray as xr

class Climdata(object):



    def __init__(self, data_path, bbox=None, hour=None):
        self.data_path = data_path
        self.bbox = bbox
        self.hour = hour

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
        if self.bbox:
            dataset = self.spatial_subset(dataset, self.bbox)
        if self.hour:
            dataset = self.time_subset(dataset, self.hour)
 
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

    def wind_speed(self):
        dataset_uv = self.read_dataset('165_166')
        if self.bbox:
            dataset_uv = self.spatial_subset(dataset_uv, self.bbox)
        dataset_uv = self.time_subset(dataset_uv, 5)
        wind_speed = np.sqrt(dataset_uv['u10']**2 + dataset_uv['v10']**2)
        return wind_speed

    def to_csv(self):
        wind_speed = self.wind_speed()
        temperature = self.read_dataset(167)

        rad = self.read_dataset(169)







if __name__ == '__main__':
    data_path = '/home/tadas/tofewsi/data/'

    #Riau bbox
    bbox = [3, -2, 99, 104]
    hour = 5
    ds = Climdata(data_path, bbox=bbox, hour=hour)

