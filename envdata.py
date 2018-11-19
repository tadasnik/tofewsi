import abc
import os
import datetime
import numpy as np
import xarray as xr

class Envdata(metaclass=abc.ABCMeta):
    def __init__(self, data_path, bbox=None, hour=None):
        self.data_path = data_path
        self.bbox = bbox
        self.hour = hour
        self.region_bbox = {'indonesia': [8.0, 93.0, -13.0, 143.0],
                                 'riau': [3, -2, 99, 104]}

    def read_hdf4(self, file_name, dataset=None):
        """
        Reads Scientific Data Set(s) stored in a HDF-EOS (HDF4) file
        defined by the file_name argument. Returns SDS(s) given
        name string provided by dataset argument. If
        no dataset is given, the function returns pyhdf
        SD instance of the HDF-EOS file open in read mode.
        """
        dataset_path = os.path.join(self.data_path, file_name)
        try:
            product = SD(dataset_path)
            if dataset == 'all':
                dataset = list(product.datasets().keys())
            if isinstance(dataset, list):
                datasetList = []
                for sds in dataset:
                    selection = product.select(sds).get()
                    datasetList.append(selection)
                return datasetList
            elif dataset:
                selection = product.select(dataset).get()
                return selection
            return product
        except IOError as exc:
            print('Could not read dataset {0}'.format(file_name))
            raise

    def read_geotif(self, file_name):
        product = xr.open_rasterio(file_name)
        return product

    def read_dataset(self, file_name):
        """
        Reads netCDF dataset using xarray.
        Args:
            parameter - (int) grib_id of the dataset to read.
        Returns
            xarray dataframe
        """
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
        lat_name = [x for x in list(dataset.coords) if 'lat' in x]
        lon_name = [x for x in list(dataset.coords) if 'lon' in x]
        dataset = dataset.where((dataset[lat_name[0]] <= bbox[0]) &
                                (dataset[lat_name[0]] >= bbox[1]), drop=True)
        dataset = dataset.where((dataset[lon_name[0]] >= bbox[2]) &
                                (dataset[lon_name[0]] <= bbox[3]), drop=True)
        return dataset

    def time_subset(self, dataset, hour=None, start_date=None, end_date=None):
        """
        Selects data within spatial bbox.
        Args:
            dataset - xarray dataset
            hour - (int) hour
        Returns:
            xarray dataset
        """
        if hour:
            dataset = dataset.sel(time=datetime.time(hour))
        return dataset

    def write_csv(self, dfr, fname):
        print('writing dataframe to csv file {0}'.format(fname))
        dfr.to_csv(fname, index=False, float_format='%.2f')
        print('finished writing')

if __name__ == '__main__':
    data_path = '/mnt/data/SEAS5'
    store_name = os.path.join(data_path, '2013-05-01_164.128_165.128_166.128_167.128_168.128_169.128_228.128_0.25deg.nc')
    fo = Envdata(data_path, os.path.join(data_path, store_name))
