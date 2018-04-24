# !/usr/bin/env python3
import datetime
import os, glob
import time
import tarfile, gzip
import shutil
import errno
import subprocess
import numpy as np
import pandas as pd
from ecmwfapi import ECMWFService

def createDir(directory):
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


def retrieveMars(dataPath, marsDict):
    """Retrieves Mars datasets

    Arguments:
        dataPath :

    Returns:
        dataFile :
    """
    server = ECMWFService("mars")
    dataFile = os.path.join(dataPath, "{0}_{1}_{2}deg.nc".format(
                                      '_'.join(marsDict['date'].split('/')),
                                      '_'.join(marsDict['param'].split('/')),
                                      marsDict['grid'].split('/')[0]))
    createDir(dataPath)
    server.execute(marsDict, dataFile)
    return dataFile


def getMarsData(start_date, end_date, dataPath, marsDict, frequency, expv):
    """Retrieves and stores datasets from Mars"""

    steps = None
    if 'step' in marsDict:
        steps = marsDict['step']
    else:
        steps = None
    dataFile = retrieveMars(dataPath, marsDict)

def join_values(values):
    """If values is a list, joins values with "/" and returns
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
    
def SEAS5_data(data_path, start_date, end_date, source_type, times, parameters, coord_bounds, step):
    """A generic method to retrieve ERA5 data from MARS"""
    if start_date != end_date:
        daterange = "{0}-{1:0>2}-{2:0>2}/to/{3}-{4:0>2}-{5:0>2}".format(start_date.year,
                                                                        start_date.month,
                                                                        start_date.day,
                                                                        end_date.year,
                                                                        end_date.month,
                                                                        end_date.day)
    else:
        daterange = '{0}-{1:0>2}-{2:0>2}'.format(start_date.year, 
                                                 start_date.month,
                                                 start_date.day)
    #Format the arguments
    bbox = join_values(coord_bounds)
    param = join_values(parameters)
    time = join_values(times)
    number = "0/to/24"
    mars_dict = {"class": "od",
                  "date": daterange,
                "expver": "1",
               "levtype": "sfc",
                "method": "1",
                "number": number,
                 "param": param,
                "origin": "ecmf",
                "stream": "mmsf",
                "system": "5",
                  "time": time,
                  "type": source_type,
                  "area": bbox,
                  "grid": "0.25/0.25",
                "format": "netcdf"}
    if step:
        step = join_values(step)
        mars_dict["step"] = step
    print(mars_dict)
    getMarsData(start_date, end_date, data_path, mars_dict, '1D', 'test')

def ERA5_data(data_path, start_date, end_date, source_type, times, parameters, coord_bounds, step):
    """A generic method to retrieve ERA5 data from MARS"""

    daterange = "{0}-{1:0>2}-{2:0>2}/to/{3}-{4:0>2}-{5:0>2}".format(start_date.year,
                                                                    start_date.month,
                                                                    start_date.day,
                                                                    end_date.year,
                                                                    end_date.month,
                                                                    end_date.day)
    #Format the arguments
    bbox = join_values(coord_bounds)
    param = join_values(parameters)
    time = join_values(times)

    mars_dict = {"class": "ea",
               "dataset": "era5",
                  "date": daterange,
                "expver": "1",
               "levtype": "sfc",
                 "param": param,
                "stream": "oper",
                  "time": time,
                  "type": source_type,
                  "area": bbox,
                  "grid": "0.25/0.25",
                "format": "netcdf"}
    if step:
        step = join_values(step)
        mars_dict["step"] = step
    print(mars_dict)
    getMarsData(start_date, end_date, data_path, mars_dict, '1D', 'test')


if __name__ == '__main__':

    # Change these as needed
    # Change this directory
    data_path = '/home/tadas/tofewsi/data/'

    Indonesia_bb = [5.47982086834, 95.2930261576, -10.3599874813, 141.03385176]
    Indonesia_bb_rounded = [8.0, 93.0, -13.0, 143.0]
    coord_bounds = Indonesia_bb_rounded

    #for 2 metre temperature, 2 metre dewpoint temperature and the 
    #wind speed components U and V we can use analysis, setting source_type to "an".
    #source_type = "an"

    # Change these as needed
    """
    for year in [2010, 2011, 2012, 2013, 2014]:
        #start_date = datetime.datetime(year - 1, 12, 31)
        #end_date = datetime.datetime(year, 12, 31)
        # setting source_type to "fc".
        #source_type = "an"
        #ERA5_data(data_path, start_date, end_date, source_type, [6, 18], ['169.128', '228.128'], coord_bounds, step=list(range(1, 13, 1)))
        #ERA5_data(data_path, start_date, end_date, source_type, list(range(24)), ['165.128', '166.128', '167.128', '168.128'], coord_bounds, step=None)



        start_date = datetime.datetime(year, 1, 1)
        end_date = datetime.datetime(year, 12, 31)
        # setting source_type to "fc".
        source_type = "an"
        #ERA5_data(data_path, start_date, end_date, source_type, [6, 18], ['169.128', '228.128'], coord_bounds, step=list(range(1, 13, 1)))
        #ERA5_data(data_path, start_date, end_date, source_type, list(range(24)), ['165.128', '166.128', '167.128', '168.128'], coord_bounds, step=None)



        data_path = '/home/tadas/atmos/data'
        ERA5_data(data_path, start_date, end_date, source_type, list(range(24)), ['164.128', '165.128', '166.128', '167.128', '168.128'], coord_bounds, step=None)
    """

    #ERA5_data(data_path, start_date, end_date, source_type, list(range(24)), ['165.128', '166.128', '167.128', '168.128'], coord_bounds, step=None)

    # Accumulated parameters 
    # In ERA5, the short forecast accumulations are accumulated from the end 
    # of the previous step.
    # Accumulated parameters are not available from the analyses.

    # for precipitation and surface solar radiation downwards we need to use forecasts,
    # setting source_type to "fc".
    # source_type = "fc"

    #ERA5_data(data_path, start_date, end_date, source_type, list(range(24)), ['165.128', '166.128', '167.128', '168.128'], coord_bounds, step=None)

    # Accumulated parameters 
    # In ERA5, the short forecast accumulations are accumulated from the end 
    # of the previous step.
    # Accumulated parameters are not available from the analyses.

    # for precipitation and surface solar radiation downwards we need to use forecasts,
    # setting source_type to "fc".
    # source_type = "fc"

    # ERA5_data(data_path, start_date, end_date, source_type, [6, 18], ['169.128', '228.128'], coord_bounds, step=list(range(1, 13, 1)))


    #MARS coordinates format 'area: North/West/South/East'

    # surface solar radiation downwards: 169.128
    # 2 metre temperature: 167.128
    # 2 metre dewpoint temperature: 168.128
    # 10 metre wind U component: 165.128
    # 10 metre wind V component: 166.128
    # total precipitaion: 228.128

    # the bellow will retrieve temperature data at 1 deg resolution
    #resolution = "1/1"
    #param = "167.128"
    #getMarsGeneric(dataPath, start_date, end_date, param, resolution, expv)

    #uncomment to retrieve wind data and compute speed
    #getWindSpeed(dataPath, start_date, end_date, expv)

    #uncomment to retrieve total precipitation and compute 24 Hour total
    #getTp24Hours(dataPath, start_date, end_date, expv)

    #uncomment to retrieve geff datasets fwi and dc.
    #getGeff(dataPath, start_date, end_date, expv)

    #SEAS5 MARS retrieval example:
    year = 2013
    start_date = datetime.datetime(year, 5, 1)
    end_date = datetime.datetime(year, 5, 1)
    data_path = '/mnt/data/SEAS5'
    source_type = 'fc'
    param_list = ['164.128', '165.128', '166.128', '167.128', '168.128', '169.128', '228.128']
    SEAS5_data(data_path, start_date, end_date, source_type, '00:00:00', param_list, coord_bounds, step=list(range(0, 5161, 6)))

