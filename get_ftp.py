import pandas as pd
import ftplib
import time
import os

def ftpConnect():
    """ Connects to FTP server. If connection is not possible within 10mins,
    sys.exit is made"""
    timeout = time.time() + 60 * 60
    while True:
        try:
            return ftplib.FTP(host='l5ftl01.larc.nasa.gov', user='anonymous')
        except:
            if time.time() > timeout:
                sys.exit('FTP connection failed for 10min')


def download(fileName, opath, ipath):
    """ Downloads a file from FTP server and stores it
    Arguments:
        fileName : string, the name of the file that is retrieved
        opath : string, the path under which the file is stored
        ipath : directory of the file on the FTP server
    Returns:
        None
    """
    ftp = ftpConnect()
    ftp.cwd(ipath)
    lf = open(os.path.join(opath, fileName), 'wb')
    ftp.retrbinary('RETR %s' % fileName, lf.write, 8 * 1024)
    lf.close()
    ftp.quit()

opath = '/mnt/data/mopitt'
ftp = ftplib.FTP(host='l5ftl01.larc.nasa.gov', user='anonymous')
ftp.cwd('MOPITT')
ftp.cwd('MOP03J.007')
dr = pd.date_range('2008-01-29', '2016-01-01', freq='D')
for dd in dr:
    try:
        ftp.cwd('{0}.{1:02}.{2:02}'.format(dd.year, dd.month, dd.day))
        fnames = ftp.nlst()
        for fn in fnames:
            with open(os.path.join(opath, fn), 'wb') as fl:
                ftp.retrbinary('RETR %s' % fn, fl.write, 8 * 1024)
                print('retireving {}'.format(fn))
        ftp.cwd('..')
    except:
        pass
ftp.quit()

