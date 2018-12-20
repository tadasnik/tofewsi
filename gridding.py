import numpy as np
import xarray as xr
import pandas as pd

def to_day_since(dtime_string):
    """
    Method returning day since the self base date. Takes string datetime in
    YYYY-MM-DD format.
    """
    dtime = pd.to_datetime(dtime_string, format='%Y-%m-%d')
    return (dtime - self.basedate).days

class Gridder(object):
    def __init__(self, lats, lons):
        self.lats, self.lons= lats, lons
        self.step = self.grid_step()
        self.bbox = self.grid_bbox()
        self.grid_bins()

    def grid_step(self):
        return (self.lons[1] - self.lons[0]).values

    def grid_bbox(self):
        lat_min, lat_max = self.lats.min(), self.lats.max()
        lon_min, lon_max = self.lons.min(), self.lons.max()
        self.lat_min = lat_min - self.step * 0.5
        self.lon_min = lon_min - self.step * 0.5
        self.lat_max = lat_max + self.step * 0.5
        self.lon_max = lon_max + self.step * 0.5
        return [lat_min, lon_min, lat_min, lat_max]
        
    def grid_bins(self):
        self.lon_bins = np.arange(self.lon_min, self.lon_max, self.step)
        self.lat_bins = np.arange(self.lat_min, self.lat_max, self.step)
 
    def binning(self, lon, lat):
        """
        Get indices of the global grid bins for the longitudes and latitudes
        of observations stored in frpFrame pandas DataFrame. Must have 'lon' and 'lat'
        columns.

        Arguments:
            lon : np.array, representing unprojected longitude coordinates.
            lat : np.array, representing unprojected longitude coordinates.

        Returns:
            Raises TypeError if frpFrame is not a pandas DataFrame
            frpFrame : pandas DataFrame
                Same DataFrame with added columns storing positional indices
                in the global grid defined in grid_bins method
        """
        lonind = np.digitize(lon, self.lon_bins) - 1
        latind = np.digitize(lat, self.lat_bins) - 1
        return lonind, latind

    def to_grid(self, dfr):
        lonind, latind = self.binning(dfr['lon'].values, dfr['lat'].values)
        dfr.loc[:, 'lonind'] = lonind
        dfr.loc[:, 'latind'] = latind
        gridded = np.zeros((self.lat_bins.shape[0],
                         self.lon_bins.shape[0]))
        grouped = pd.DataFrame({'count' : dfr.groupby(['lonind', 'latind']).size()}).reset_index()
        latinds = grouped['latind'].values.astype(int)
        loninds = grouped['lonind'].values.astype(int)
        gridded[latinds, loninds] = grouped['count'].astype(int)
        gridded = np.flipud(gridded)
        return gridded

    def to_xarray(self, data, var_name, timestamps):
        lats = np.arange((-90 + self.step / 2.), 90., self.step)[::-1]
        lons = np.arange((-180 + self.step / 2.), 180., self.step)
        dataset = xr.Dataset({var_name: (['latitude', 'longitude', 'date'], data)},
                              coords={'latitude': lats,
                                     'longitude': lons,
                                     'date': timestamps})
        return dataset

    def grid_dfr(self, dfr):
        dates = pd.date_range(dfr.date.min(), dfr.date.max(), freq='D')
        lonind, latind = self.binning(dfr['lon'].values, dfr['lat'].values)
        dfr.loc[:, 'lonind'] = lonind
        dfr.loc[:, 'latind'] = latind
        dfa = pd.DataFrame({'count': dfr.groupby(['date', 'latind', 'lonind']).size()})
        dfa = dfa.reset_index()
        dfa.loc[:, 'dind'] = (dfa.date - dfa.date.min()).dt.days
        grids = np.zeros((self.lats.shape[0], self.lons.shape[0], dfa.dind.max() + 1),
                         dtype = int)
        grids[dfa.latind, dfa.lonind, dfa.dind] = dfa['count'].astype(int)
        grids = np.flip(grids, axis = 0)
        dataset = xr.Dataset({'count': (['latitude', 'longitude', 'time'], grids)},
                              coords={'latitude': self.lats,
                                      'longitude': self.lons,
                                      'time': dates})
        return dataset


    def grid_centroids(self, years, dfr_list, distance):
            dsy = []
            for nr, dur in enumerate(['2', '4', '8', '16']):
                dfr = dfr_list[nr]
                gr_year = dfr[dfr.year == year]
                grouped_days = gr_year.groupby('date')
                grids = [self.to_grid(x[1]) for x in grouped_days]
                timestamps = [x[0] for x in grouped_days]
                netcdf_store = self.to_xarray(np.dstack(grids), 'ign_agg_{0}'.format(dur), timestamps)
                dsy.append(netcdf_store)
            dsa = xr.merge(dsy)
            dsa.to_netcdf('/mnt/data/frp/ignitions_tropics_{0}_{1}_frp.nc'.format(year, distance),
                          encoding={'ign_agg_2': {'dtype': 'int16', 'zlib': True},
                                    'ign_agg_4': {'dtype': 'int16', 'zlib': True},
                                    'ign_agg_8': {'dtype': 'int16', 'zlib': True},
                                    'ign_agg_16': {'dtype': 'int16', 'zlib': True}})

if __name__ == '__main__':
    """
    fwi = xr.open_dataset('/home/tadas/data/fwi/fwi_arr.nc')
    dfr = pd.read_parquet('/home/tadas/data/frp/M6_indonesia.parquet')
    gri = Gridder(fwi.latitude, fwi.longitude)
    ds = gri.grid_dfr(dfr)
    """

