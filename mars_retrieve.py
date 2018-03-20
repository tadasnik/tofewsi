# !/usr/bin/env python
import datetime
import os, glob
import time
import tarfile, gzip
import shutil
import errno
import subprocess
import numpy as np
import firegrib as fg
import pandas as pd

from ecmwfapi import ECMWFService, ECMWFDataServer
import utilities as ut

def getMonthlyDateRanges(start_date, end_date):
    stDates = pd.date_range(start=start_date, end=end_date, freq='MS')
    stDates.freq=None
    stDates = stDates.insert(0, start_date)
    enDates = pd.date_range(start=start_date, end=end_date, freq='M')
    stDates.freq=None
    enDates = enDates.append(pd.Index([end_date]))
    return stDates, enDates

def populateGeff(dataFile, start_date, end_date, expv):
    """Store geff fwi and dc datasets"""
    dateRange = pd.date_range(start=start_date,
                               end=end_date, freq='D')
    try:
        tar = tarfile.open(os.path.basename(dataFile))
        tar.extractall()
        tar.close()
    except:
        print 'Could not untar geff archive'
        sys.exit()
    daypaths = [day.strftime('%Y%m%d') for day in dateRange.date]
    expvdir = str(expv).zfill(4)
    for date in dateRange:
        fwiName = glob.glob('*fwi_{0}*'.format(date.strftime('%Y%m%d')))[0]
        dcName = glob.glob('*dc_{0}*'.format(date.strftime('%Y%m%d')))[0]
        fwiOutName = os.path.join(expvdir, date.strftime('%Y%m%d'), '0000', 'fwi.nc')
        dcOutName = os.path.join(expvdir, date.strftime('%Y%m%d'), '0000', 'dc.nc')
        with gzip.open(fwiName, 'rb') as f_in, open(fwiOutName, 'wb') as f_out:
             shutil.copyfileobj(f_in, f_out)
        with gzip.open(dcName, 'rb') as f_in, open(dcOutName, 'wb') as f_out:
             shutil.copyfileobj(f_in, f_out)
    for archFile in glob.glob('*.nc.gz'):
        os.remove(archFile)

def retrieveMars(dataPath, marsDict):
    """Retrieves Mars datasets

    Arguments:
        dataPath :

    Returns:
        dataFile :
    """
    server = ECMWFService("mars")
    dataFile = os.path.join(dataPath, "{0}_{2}deg_tmp.nc".format(
                                      marsDict['stream'],
                                      marsDict['grid'].split('/')[0]))
    ut.createDir(dataPath)
    server.execute(marsDict, dataFile)
    return dataFile

def create_directories(start_date, end_date, dataPath, expv, frequency, steps):
    """creates the directories in which the dataFile will be stored

    Arguments:
        start_date :
        end_date :
        dataPath :
        expv :

    Returns:
        None
        Raises an exception if the path could not be created
    """
    # create date_range from start_date to end_date and create directories for each step
    date_range = pd.date_range(start=start_date,
                               end=end_date + datetime.timedelta(hours=23), freq=frequency)
    daypaths = [day.strftime('%Y%m%d') for day in date_range.date]
    expvdir = str(expv).zfill(4)
    for dpath in daypaths:
        ut.createDir(os.path.join(dataPath, expvdir, dpath))
        for hour in np.unique(date_range.hour):
            hourpath = str(hour).zfill(2).ljust(4, '0')
            ut.createDir(os.path.join(dataPath, expvdir, dpath, hourpath))
            if steps:
                for step in steps.split('/'):
                    ut.createDir(os.path.join(dataPath, expvdir, dpath, hourpath, step))


def populate_directories(dataFile, expvdir, steps):
    """stores datasets in the directory structure created by 
    create_directories function.

    Arguments:
        dataFile :
        expvdir :

    Returns:
        None
        Raises an exception if the path could not be created
    """
    curdir = os.getcwd()
    # try to change to directory of dataFile
    try:
        os.chdir(os.path.dirname(dataFile))
        if steps:
            subprocess.Popen(['grib_copy', '-v', os.path.split(dataFile)[1],
                              expvdir + '/[date]/[time]/[step]/[paramId].grb'])
        else:
            subprocess.Popen(['grib_copy', '-v', os.path.split(dataFile)[1],
                              expvdir + '/[date]/[time]/[paramId].grb'])
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    finally:
        # switch back to original directory
        os.chdir(curdir)

def tp24Hours(dataPath, start_date, end_date, expv):
    """Calculates and stores total precipitation during previous 24 hours
    from era import interim reanalysis forecasts"""
    dayRange = pd.date_range(start=start_date,
                               end=end_date + datetime.timedelta(hours=23), freq='D')
    #dayPaths = [day.strftime('%Y%m%d') for day in np.unique(dateRange.date)]
    expvdir = str(expv).zfill(4)
    for nr,day in enumerate(dayRange):
        #ut.createDir(os.path.join(dataPath, expvdir, daypaths[nr]))
        dayPath = os.path.join(dataPath, expvdir, day.strftime('%Y%m%d'))
        yesterday = dayRange[nr]
        yestPath = os.path.join(dataPath, expvdir, yesterday.strftime('%Y%m%d'))

        tpYest0006 = fg.Field(os.path.join(yestPath, '0000', '6', '228.grb'))
        tpYest0012 = fg.Field(os.path.join(yestPath, '0000', '12', '228.grb'))
        tpYest1206 = fg.Field(os.path.join(yestPath, '1200', '6', '228.grb'))
        tpYest1212 = fg.Field(os.path.join(yestPath, '1200', '12', '228.grb'))

        tpDay0006 = fg.Field(os.path.join(dayPath, '0000', '6', '228.grb'))
        tpDay0012 = fg.Field(os.path.join(dayPath, '0000', '12', '228.grb'))
        tpDay1206 = fg.Field(os.path.join(dayPath, '1200', '6', '228.grb'))
        tpDay1212 = fg.Field(os.path.join(dayPath, '1200', '12', '228.grb'))

        tp0000 = fg.Field(os.path.join(dayPath, '0000', '6', '228.grb'))
        tp0000.val = tpYest0012.val + tpYest1212.val
        with open(os.path.join(dayPath, '0000', 'tp24Hours.grb'), 'w') as fp:
            tp0000.streamout(fp)

        tp0600 = fg.Field(os.path.join(dayPath, '0000', '6', '228.grb'))
        tp0600.val = (tpYest0012.val - tpYest0006.val) + tpYest1212.val + tpDay0006.val 
        with open(os.path.join(dayPath, '0600', 'tp24Hours.grb'), 'w') as fp:
            tp0600.streamout(fp)

        tp1200 = fg.Field(os.path.join(dayPath, '0000', '6', '228.grb'))
        tp1200.val = tpYest1212.val + tpDay0012.val
        with open(os.path.join(dayPath, '1200', 'tp24Hours.grb'), 'w') as fp:
            tp1200.streamout(fp)

        tp1800 = fg.Field(os.path.join(dayPath, '0000', '6', '228.grb'))
        tp1800.val = (tpYest1212.val - tpYest1206.val) + tpDay0012.val + tpDay1206.val
        with open(os.path.join(dayPath, '1800', 'tp24Hours.grb'), 'w') as fp:
            tp1800.streamout(fp)

def windSpeed(dataPath, start_date, end_date, expv):
    """Calculates and stores wind speed form v and u wind components"""
    dateRange = pd.date_range(start=start_date,
                               end=end_date + datetime.timedelta(hours=23), freq='6H')
    #dayPaths = [day.strftime('%Y%m%d') for day in np.unique(dateRange.date)]
    expvdir = str(expv).zfill(4)
    for nr,date in enumerate(dateRange):
        #ut.createDir(os.path.join(dataPath, expvdir, daypaths[nr]))
        datePath = os.path.join(dataPath, expvdir, date.strftime('%Y%m%d'),
                                        str(date.hour).zfill(2).ljust(4,'0'))

        windU = fg.Field(os.path.join(datePath, '165.grb'))
        windV = fg.Field(os.path.join(datePath, '166.grb'))
        windField = fg.Field(os.path.join(datePath, '165.grb'))
        windField.val = np.sqrt(windU.val**2 + windV.val**2)
        with open(os.path.join(datePath, 'windSpeed.grb'), 'w') as fp:
            windField.streamout(fp)


def getMarsData(start_date, end_date, dataPath, marsDict, frequency, expv):
    """Retireves and stores datasets from Mars"""

    steps = None
    if 'step' in marsDict:
        steps = marsDict['step']
    else:
        steps = None
    dataFile = retrieveMars(dataPath, marsDict)
    #create_directories(start_date, end_date, dataPath, expv, frequency, steps)
    #populate_directories(dataFile, expv, steps)

def getDataRange(start_date, end_date):
    return "{0}-{1:0>2}-{2:0>2}/to/{3}-{4:0>2}-{5:0>2}".format(start_date.year,
                                                               start_date.month,
                                                               start_date.day,
                                                               end_date.year,
                                                               end_date.month,
                                                               end_date.day)
 
def getGeff(dataPath, start_date, end_date, expv):
    """Retrieves and stores geff fwi and dc products"""
    dataFile = "fwi_dc.tar"
    stDates, enDates = getMonthlyDateRanges(start_date, end_date)
    for dateR in zip(stDates, enDates):
        geffDict = {
            "dataset": "geff_reanalysis",
            "date": getDataRange(dateR[0], dateR[1]),
            "origin": "fwis",
            "param": "dc/fwi",
            "step": "00",
            "time": "0000",
            "type": "an",
            "target": dataFile
        }
        curdir = os.getcwd()
        ut.createDir(dataPath)
        os.chdir(dataPath)
        server = ECMWFDataServer()
        server.retrieve(geffDict)
        create_directories(dateR[0], dateR[1], dataPath, expv, 'D', None)
        populateGeff(dataFile, dateR[0], dateR[1], expv)
        os.chdir(curdir)

def getMarsGeneric(dataPath, start_date, end_date, param, resolution, expv):
    expv = 'oper'
    stDates, enDates = getMonthlyDateRanges(start_date, end_date)
    for dateR in zip(stDates, enDates):
        marsDict = {
            "class": "od",
            "stream": "oper",
            "expver": "1",
            "type": "an",
            "levtype": "sfc",
            "param": param,
            "date": getDataRange(dateR[0], dateR[1]),
            "time": "0000/0600/1200/1800",
            "grid": resolution
            }
        getMarsData(dateR[0], dateR[1], dataPath, marsDict, '6H', expv)

def getTp24Hours(dataPath, start_date, end_date, expv):
    expv = 'oper'
    tpstart_date = start_date - datetime.timedelta(days=1)
    stDates, enDates = getMonthlyDateRanges(tpstart_date, end_date)
    for dateR in zip(stDates.unique(), enDates):
        print dateR
        tpDict = {
            'class': "ei",
            'stream': "oper",
            'expver': "1",
            'type': "fc",
            'levtype': "sfc",
            'param': "228.128",
            "date": getDataRange(dateR[0], dateR[1]),
            'time': "0000/1200",
            'step': "6/12",
            'grid': "1/1"
        }
        getMarsData(dateR[0], dateR[1], dataPath, tpDict, '6H', expv)
        time.sleep(10)
        tp24Hours(dataPath, dateR[0], dateR[1], expv)

def getWindSpeed(dataPath, start_date, end_date, expv):
    # wind speed
    expv = 'oper'
    stDates, enDates = getMonthlyDateRanges(start_date, end_date)
    for dateR in zip(stDates, enDates):
        wsDict = {
            'class': "od",
            'dataset': "interim",
            'stream': "oper",
            'expver': "1",
            'type': "an",
            'levtype': "sfc",
            'param': "165.128/166.128",
            'date': getDataRange(dateR[0], dateR[1]),
            'time': "0000/0600/1200/1800",
            'grid': "1/1"
        }
        getMarsData(dateR[0], dateR[1], dataPath, wsDict, '6H', expv)
        time.sleep(10)
        windSpeed(dataPath, dateR[0], dateR[1], expv)

def ERA5_surface_rad(data_path, start_date, end_date, coord_bounds):
    daterange = "{0}-{1:0>2}-{2:0>2}/to/{3}-{4:0>2}-{5:0>2}".format(start_date.year,
                                                                    start_date.month,
                                                                    start_date.day,
                                                                    end_date.year,
                                                                    end_date.month,
                                                                    end_date.day)
    bbox = '/'.join([str(x) for x in coord_bounds])
    mars_dict = {
    "class": "ea",
    "dataset": "era5",
    "date": daterange,
    "expver": "1",
    "levtype": "sfc",
    "param": "169.128",
    "step": "9/10/11",
    "stream": "oper",
    "time": "18:00:00",
    "type": "fc",
    "area": bbox,
    "grid": "0.25/0.25"
    }
    print mars_dict
    getMarsData(start_date, end_date, data_path, mars_dict, '1D', 'test')


if __name__ == '__main__':

    # Change these as needed
    start_date = datetime.datetime(2015, 1, 1)
    end_date = datetime.datetime(2015, 12, 31)

    # Change this directory
    data_path = '/home/tadas/tofewsi/data/'

    expv = 'oper'

    #ERA5_surface_rad(data_path, start_date, end_date)

    Indonesia_bb = [5.47982086834, 95.2930261576, -10.3599874813, 141.03385176]
    Indonesia_bb_rounded = [6.0, 95.0, -11.0, 142.0]
    coord_bounds = Indonesia_bb_rounded

    #MARS coordinates format 'area: North/West/South/East'
    #uncomment to retrieve total column water vapour, soil moisture or/and temperature
    # for tcwv param is "137.128",
    # for temperature param is "167.128",
    # for soil moisture param is "39.128"
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

