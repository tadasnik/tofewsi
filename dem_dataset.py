import glob
import xarray as xr
import pandas as pd

def dem_tif_to_csv(fname, dem):
    """
    Read srtm raster dataset and write to csv
    """
    x_arr = xr.open_rasterio(fname)
    dfr = x_arr.to_dataframe(name=dem)
    dfr.reset_index(inplace=True)
    dfr = dfr.drop('band', axis=1)
    cols = ['lat', 'long', dem]
    dfr.columns = cols
    dfr = dfr.round({'lat': 3,
                     'long': 3,
                     dem: 3})
    if dem == 'elevation':
        dfr['elevaion'] = dfr['elevation'].astype(int)
    elif dem == 'aspect':
        dfr['aspect'][dfr['aspect'] < 0] = 0


    out_fname = 'data/riau_{}_5km.csv'.format(dem)
    print('writing dataframe to csv file {0}'.format(out_fname))
    dfr.to_csv(out_fname, index=False)
    print('finished writing')

if __name__ == '__main__':
    for dem in ['elevation', 'slope', 'aspect']:
        fname = glob.glob('data/riau_{0}_5km.tif'.format(dem))[0]
        dem_tif_to_csv(fname, dem)
