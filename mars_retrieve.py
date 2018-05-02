# !/usr/bin/env python3
import os
import sys
import errno
import datetime
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
    """
    A class facilitating data retrieval from MARS
    """
    def __init__(self, data_path, start_date, end_date, bbox):
        self.data_path = data_path
        self.date_range = mars_date_range(start_date, end_date)
        self.bbox = join_values(bbox)
        #MARS dictionary with items which are shared between all retrievals:
        self.mars_dict = { "date": self.date_range,
                         "expver": "1",
                        "levtype": "sfc",
                           "area": self.bbox,
                           "grid": "0.25/0.25",
                         "format": "netcdf" }
        
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
        param_list = ['164.128', '165.128', '166.128', '167.128',
                      '168.128', '169.128', '228.128']
        #Add items specific to SEAS5 to mars_dict
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

    def ERA5_mars_dict(self, source_type):
        """
        Add items to mars_dict specific to ERA5
        """
        self.mars_dict["class"] = "ea"
        self.mars_dict["dataset"] = "era5"
        self.mars_dict["stream"] = "oper"
        self.mars_dict["time"] = join_values([6, 18])
        self.mars_dict["type"] = source_type
        if source_type == "fc":
            param_list = ['169.128', '228.128']
            self.mars_dict["step"] = join_values(list(range(1, 13, 1)))
        elif source_type == "an":
            param_list = ['165.128', '166.128', '167.128', '168.128']
        else:
            print('Do not know source type {0}'.format(source_type))
            sys.exit()
        self.mars_dict["param"] = join_values(param_list)

    def retrieve_SEAS5(self):
        """
        Create SEAS5 mars dictionary and call mars
        """
        self.SEAS5_mars_dict()
        self.call_mars()

    def retrieve_ERA5(self):
        """
        Create ERA5 mars dictionary and call mars.
        Needs to be done twice, once for forecast once for analysis.
        """
        self.ERA5_mars_dict(source_type = "fc")
        self.call_mars()
        self.ERA5_mars_dict(source_type = "an")
        self.call_mars()

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
                                     '_'.join(self.mars_dict['date'].split('/')),
                                     '_'.join(self.mars_dict['param'].split('/')),
                                     self.mars_dict['grid'].split('/')[0]))
        return data_file_name
     

if __name__ == '__main__':

    # How to setup ECMWF data access:
    # https://software.ecmwf.int/wiki/display/WEBAPI/Accessing+ECMWF+data+servers+in+batch
    

    # Change these as needed
    #MARS coordinates format 'area: North/West/South/East'
    #Indonesia bounding box = [5.47982086834, 95.2930261576, -10.3599874813, 141.03385176]
    #Round Indonesia bb to get data for wider area
    coord_bounds = [8.0, 93.0, -13.0, 143.0]

    #MARS parameters
    # surface solar radiation downwards: 169.128
    # 2 metre temperature: 167.128
    # 2 metre dewpoint temperature: 168.128
    # 10 metre wind U component: 165.128
    # 10 metre wind V component: 166.128
    # total precipitaion: 228.128

    #ERA5 for 2 metre temperature, 2 metre dewpoint temperature and the 
    #wind speed components U and V we can use analysis, setting source_type to "an".
    #Accumulated fields surface solar radiation downwards and total precipitation
    #are only available as forecasts, source_type "fc"

    # ERA5 example for 2013
    """
    year = 2013
    # start_date and end_date are the same for SEAS5
    start_date = datetime.datetime(year - 1, 12, 31)
    end_date = datetime.datetime(year, 12, 31)
    data_path = '/mnt/data/era5'
    #Create a Marser class instance
    mar = Marser(data_path, start_date, end_date, coord_bounds)
    #Then we can either:
    #invoke method to fill mars dictionary for analysis fields:
    mar.ERA5_mars_dict(source_type = 'an')
    # look if it is ok
    print(mar.mars_dict)
    # if it does, incomment the line below to call ecmwf:
    # mar.call_mars()

    #invoke method to fill mars dictionary for forecast fields:
    mar.ERA5_mars_dict(source_type = 'fc')
    # look if it is ok
    print(mar.mars_dict)
    # if it does, call ecmwf:
    # if it does, incomment the line below to call ecmwf:
    #mar.call_mars()

    #or invoke ERA5 retireval method which will do all of the above in one step:
    #mar.retrieve_ERA5()
    """


    # SEAS5 example for 2013 May hindcasts
    """
    year = 2013
    # start_date and end_date are the same for SEAS5
    start_date = datetime.datetime(year, 5, 1)
    end_date = datetime.datetime(year, 5, 1)
    data_path = '/mnt/data/seas5'
    #Create a Marser class instance
    mar = Marser(data_path, start_date, end_date, coord_bounds)
    #invoke method to fill mars dictionary
    mar.SEAS5_mars_dict()
    # look if it is ok
    print(mar.mars_dict)
    # if it does, incomment the line below to call ecmwf:
    # mar.call_mars()
    """


