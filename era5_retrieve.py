# !/usr/bin/env python3
import os
import sys
import copy
import errno
import datetime
from ecmwfapi import ECMWFService
from dateutil.relativedelta import relativedelta

def create_directory(directory):
    """ Create a new directory under 'dir' if possible
    Arguments:
        directory : a string to the dir to be made
    Returns:
        None : if directory could be made or already existed
        Throws an OSError if the directory couldn't be made and didn't exist
    """
    try:
        os.makedirs(directory)
    except OSError as exc:
        # If the directory already exists, no OSError is raised, any other
        # error will raise an exception
        if directory != '' and exc.errno != errno.EEXIST:
            raise

def join_values(values):
    """
    If values is a list, joins values with "/" and returns
    the product as a single string. If values is a single value
    returns it back. Values are converted to strings.
    Args:
        values (str/list of strings)
    Returns:
        string
    """
    if isinstance(values, list):
        return '/'.join([str(x) for x in values])
    else:
        return str(values)

def mars_monthly_date_range(start_date, end_date):
    """
    Format date range string as required by MARS for monthly means retrieval.
    Args:
        start_date : datetime object
        end_date : datetime object
    Returns:
        string
    """
    monthly_dates = []
    st_date = copy.copy(start_date)
    #make sure st_date is the first of the month
    st_date.replace(day = 1)
    while st_date <= end_date:
        date_to_retrieve = '{0}{1:0>2}{2:0>2}'.format(st_date.year,
                                                      st_date.month,
                                                      st_date.day)
        monthly_dates.append(date_to_retrieve)
        st_date += relativedelta(months = 1)
    return '/'.join(monthly_dates)


def mars_date_range(start_date, end_date):
    """
    Format date range string as required by MARS. If both arguments
    are the same, date string is returned.
    Args:
        start_date : datetime object
        end_date : datetime object
    Returns:
        string
    """
    if start_date != end_date:
        date_range = '{0}-{1:0>2}-{2:0>2}/to/{3}-{4:0>2}-{5:0>2}'
        date_range = date_range.format(start_date.year,
                                       start_date.month,
                                       start_date.day,
                                       end_date.year,
                                       end_date.month,
                                       end_date.day)
    else:
        date_range = '{0}-{1:0>2}-{2:0>2}'
        date_range = date_range.format(start_date.year,
                                       start_date.month,
                                       start_date.day)
    return date_range


class Marser(object):
    """
    A class facilitating data retrieval from MARS
    """
    def __init__(self, data_path, start_date, end_date, grid, bbox = None):
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.date_range = mars_date_range(self.start_date, self.end_date)
        #MARS dictionary with items which are shared between all retrievals:
        self.mars_dict = { "date": self.date_range,
                         "expver": "1",
                        "levtype": "sfc",
                           "grid": join_values([grid, grid]),
                         "format": "netcdf"}
        if bbox:
            self.bbox = join_values(bbox)
            self.mars_dict["area"] = self.bbox
        #ECMWF parameters:
        # surface solar radiation downwards: 169.128
        # 2 metre temperature: 167.128
        # 2 metre dewpoint temperature: 168.128
        # 10 metre wind U component: 165.128
        # 10 metre wind V component: 166.128
        # total precipitaion: 228.128
        self.param_dict = { "windU": 165.128,
                            "windV": 166.128,
                             "temp": 167.128,
                           "dptemp": 168.128,
                         "solarrad": 169.128,
                          "totprep": 228.128 }

    def SEAS5_mars_dict(self):
        """
        Add items to mars_dict specific to SEAS5
        """
        #retrieve all parameters
        param_list = ['165.128', '166.128', '167.128',
                      '168.128', '169.128', '228.128']
        #Add items specific to SEAS5 to mars_dict
        self.mars_dict["date"] = mars_date_range(self.start_date, self.end_date)
        self.mars_dict["class"] = "od"
        self.mars_dict["method"] = "1"
        self.mars_dict["number"] = "0/to/24"
        self.mars_dict["param"] = join_values(param_list)
        self.mars_dict["origin"] = "ecmf"
        self.mars_dict["stream"] = "mmsf"
        self.mars_dict["system"] = "5"
        self.mars_dict["time"] = "00:00:00"
        self.mars_dict["type"] = "fc"
        self.mars_dict["step"] = join_values(list(range(0, 5161, 6)))

    def ERA5_mars_dict(self, stream, param_list, times, source_type):
        """
        Add items to mars_dict specific to ERA5
        """
        self.mars_dict["class"] = "ea"
        self.mars_dict["dataset"] = "era5"
        self.mars_dict["stream"] = stream
        self.mars_dict["time"] = times
        if stream != "moda":
            self.mars_dict["date"] = mars_date_range(self.start_date, self.end_date)
        else:
            self.mars_dict["date"] = mars_monthly_date_range(self.start_date, self.end_date)
        self.mars_dict["type"] = source_type
        if source_type == "fc" and stream == "oper":
            self.mars_dict["step"] = join_values(list(range(1, 13, 1)))
        self.mars_dict["param"] = join_values(param_list)

    def call_mars(self):
        """
        Retrieves Mars datasets

        Arguments:
            data_path:
            :
        """
        print('Calling MARS with dictionary:\n {0}'.format(self.mars_dict))
        server = ECMWFService("mars")
        data_file_name = self.get_file_name()
        create_directory(self.data_path)
        server.execute(self.mars_dict, data_file_name)

    def get_file_name(self):
        """
        Returns absolute path file name for mars data.
        """
        data_file_name= os.path.join(self.data_path, "{0}_{1}_{2}deg.nc".format(
                                     '_'.join([self.mars_dict['date'].split('/')[0],
                                              self.mars_dict['date'].split('/')[-1]]),
                                     '_'.join(self.mars_dict['param'].split('/')),
                                     self.mars_dict['grid'].split('/')[0]))
        return data_file_name

if __name__ == '__main__':

    # How to setup ECMWF data access:
    # https://software.ecmwf.int/wiki/display/WEBAPI/Accessing+ECMWF+data+servers+in+batch

    # Change these as needed
    #grid spacing in degrees
    grid = 0.25

    #MARS coordinates format 'area: North/West/South/East'
    #Indonesia bounding box = [5.47982086834, 95.2930261576, -10.3599874813, 141.03385176]
    #Round Indonesia bb to get data for wider area
    bbox = [8.0, 93.0, -13.0, 143.0]

    #MARS parameters
    # surface solar radiation downwards: 169.128
    # 2 metre temperature: 167.128
    # 2 metre dewpoint temperature: 168.128
    # 10 metre wind U component: 165.128
    # 10 metre wind V component: 166.128
    # total precipitaion: 228.128

    #ERA5 for 2 metre temperature, 2 metre dewpoint temperature and the 
    #wind speed components U and V we can use analysis, setting source_type to "an".
    #Precipitation is only available as forecasts, hence source_type must be "fc"

    #change this!!!!

    #Accumulated fields surface solar radiation downwards and total precipitation
    #are only available as forecasts, source_type "fc"

    # In this example era5 monthly means for two decades (2000s and 2010s) will be retrieved
    # At the moment data for 2000s is available only starting from 2008! End date is Feb 2018,
    # as of 21/05/2018

    #Europe bbox:
    #bbox = [59, -10, 34, 30]

    # The following runs retrieval
    #data_path = '/mnt/data/era5/indonesia'

    # invoke ERA5 dictionary filling method to retireve monthly mean 2m temperature
    #mars.ERA5_mars_dict(stream = "moda", param_list = ['167.128'], source_type = "an")

    #era5 hourly.
    for year in [2009, 2010, 2011, 2012, 2013, 2014, 2015]:
        start_date = datetime.datetime(year, 1, 1)
        end_date = datetime.datetime(year, 12, 31)
        # Instantiate Mars object with defined properties
        mars = Marser(data_path, start_date, end_date, grid, bbox = bbox)
        #first analysis fields:
        times = list(range(24))
        param_list = ['165.128', '166.128', '167.128', '168.128']
        mars.ERA5_mars_dict(stream = "oper", param_list = param_list, times = times, source_type = "an")
        mars.call_mars()
        # call ecmwf to retrieve the data
        #mars.call_mars()
        #Total radiation downwards and precipitation
        #are stored as forecasts, hence separate retrieval.
        #times = [6, 18]
        #mars.ERA5_mars_dict(stream = "oper", param_list = ['169.128', '228.128'], times=times, source_type = "fc")
        #mars.call_mars()

        # we can check what mars dictionary looks like:
        #print(mars.mars_dict)
        # If it looks reasonable, 
        # call ecmwf to retrieve the data

