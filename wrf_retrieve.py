import os
import datetime
import pandas as pd
from ecmwfapi import ECMWFService

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
    def __init__(self, data_path, start_date, end_date, bbox = None, grid = None):
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.date_range = mars_date_range(self.start_date, self.end_date)
        self.bbox = bbox
        self.grid = grid

    def ERA5_mars_dict(self, param_list, levtype):
        """
        Creates dictionary with parameters needed to call Mars
        """
        self.mars_dict = {"date": self.date_range,
                          "expver": "1",
                          "levtype": levtype,
                          "class": "ea",
                          "dataset": "era5",
                          "stream": "oper",
                          "time": join_values(list(range(0, 22, 3))),
                          "param": param_list,
                          "type": "an"}
        if levtype == "ml":
            self.mars_dict['levelist'] = join_values(list(range(1, 138, 1)))
        if self.bbox:
            self.bbox = join_values(bbox)
            self.mars_dict["area"] = self.bbox
            self.mars_dict["format"] = "netcdf"
            self.mars_dict["grid"] = join_values([self.grid, self.grid])

    def call_mars(self):
        """
        Retrieves Mars datasets

        Arguments:
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
        data_file_name= os.path.join(self.data_path, "{0}_to_{1}_{2}".format(
                                     self.mars_dict['date'].split('/')[0],
                                     self.mars_dict['date'].split('/')[-1],
                                     self.mars_dict['levtype']))
        return data_file_name

if __name__ == '__main__':
    #indonesia bbox
    bbox = [6, 96, -4, 109]

    #mars parameter list levels
    param_list_levels = ["129/130/131/132/133/135/152/248"]

    #mars parameter list surface
    param_list_sfc = ["34.128/39.128/40.128/41.128/42.128/" +
                      "66.128/67.128/139.128/151.128/159.128/" +
                      "164.128/165.128/166.128/167.128/168.128/" +
                      "170.128/172.128/183.128/235.128/236.128/246.228/247.228"]

    #where data will be stored
    data_path = ''


    year = 2008
    #If only one day of data:
    #start_date = datetime.datetime(year, 1, 1)
    #end_date = datetime.datetime(year, 2, 1)

    #if for whole year:
    start_date = datetime.datetime(year, 1, 1)
    end_date = datetime.datetime(year, 12, 31)

    # Instantiate Mars object with defined properties
    # If provide bbox and grid will retrieve data only for area defined by bbox
    mars = Marser(data_path, start_date, end_date, bbox = bbox, grid = "0.25")

    #Replace previous line with this will get global. Change the date range as well.
    #mars = Marser(data_path, start_date, end_date)

    #first level parameters:
    mars.ERA5_mars_dict(param_list=param_list_levels, levtype="ml")
    # call ecmwf to retrieve the data
    mars.call_mars()

    #and surface parameters
    mars.ERA5_mars_dict(param_list=param_list_sfc, levtype="sfc")
    # call ecmwf to retrieve the data
    mars.call_mars()

