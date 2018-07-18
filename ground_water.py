import xarray as xr

#Coordinates for the sites (lon, lat)
coords = {'acacia': [101.49, 1.3], 
          'forest': [101.4, 1.27],
          'rubber': [101.44, 1.39]}

#date renges
times = {'acacia': pd.date_range[, 1.3], 
         'forest': [101.4, 1.27],
         'rubber': [101.44, 1.39]}



fname = '/mnt/data'
dataset = xr.open_dataset(fname)

acacia = dataset.sel(longitude=101.5)
acacia = acacia.sel(latitude=1.25)

forest = dataset.sel(longitude=101.5)
forest = forest.sel(latitude=1.25)

rubber = dataset.sel(longitude=101.5)
rubber = rubber.sel(latitude=1.5)

acacia = dataset.where((dataset['longitude'] == 101.5) & (dataset['latitude'] == 1.25), drop=True)
acacia = acacia.to_dataframe()
