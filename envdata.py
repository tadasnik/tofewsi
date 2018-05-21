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

    @abc.abstractmethod
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
 
    def read_soil_grids_tiff(self, sp_res=0.05):
        """
        Read SoilGrids datasets stored in GeoTiff
        """
        datasets = []
        for prod_key in self.codes_names:
            products = []
            for lev_key in self.levels:
                fname = '{0}_M_{1}_5km_ll.tif'.format(prod_key, lev_key)
                fname_path = os.path.join(self.data_path, fname)
                product = xr.open_rasterio(fname_path)
                products.append(product.values.squeeze())
            lons = product.x.values
            #latitudes need tiying as they are off by a bit
            #porbably due to float conversion somewhere in the pipline
            lats = np.arange(product.y[-1].values.round(0) + sp_res/2, 
                             product.y[0].values + sp_res/2, 
                             sp_res)[::-1]
            dataset = xr.Dataset({self.codes_names[prod_key]: (('level', 'latitude', 'longitude'),
                                                                np.array(products))},
                                  coords = {'level': list(self.levels.values()),
                                         'latitude': lats,
                                        'longitude': lons})
            datasets.append(dataset)
        dataset = xr.merge(datasets)
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
        dataset = dataset.where((dataset[lat_name[0]] < bbox[0]) &
                                (dataset[lat_name[0]] > bbox[1]), drop=True)
        dataset = dataset.where((dataset[lon_name[0]] > bbox[2]) &
                                (dataset[lon_name[0]] < bbox[3]), drop=True)
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

