import cdsapi

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

ind_area = join_values([8.0, 93.0, -13.0, 143.0])

c = cdsapi.Client()
hincast_years = [str(year) for year in range(1993, 2019, 1)]
lead_hours = [str(x) for x in range(6, 5161, 6)]

"""
for year in range(2019, 2020, 1):
    for month in range(12, 13, 1):
        print(year, month)
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type':'reanalysis',
                'format':'netcdf',
                'variable':[
                '10m_u_component_of_wind', '10m_v_component_of_wind',
                '2m_dewpoint_temperature',
                '2m_temperature',
                'total_precipitation', 'total_cloud_cover'
                ],
                'year':'{0}'.format(year),
                'month':['{0}'.format(month)],
                'day':[
                    '1','2','3',
                    '4','5','6',
                    '7','8','9',
                    '10','11','12',
                    '13','14','15',
                    '16','17','18',
                    '19','20','21',
                    '22','23','24',
                    '25','26','27',
                    '28','29','30',
                    '31',
                ],
                'time':[
                    '00:00','01:00','02:00',
                    '03:00','04:00','05:00',
                    '06:00','07:00','08:00',
                    '09:00','10:00','11:00',
                    '12:00','13:00','14:00',
                    '15:00','16:00','17:00',
                    '18:00','19:00','20:00',
                    '21:00','22:00','23:00',
                ]
            },
            '/mnt/data/era5/glob/{0}_{1}.nc'.format(year, month))

c.retrieve(
    'seasonal-monthly-single-levels',
    {
        'format':'grib',
        'originating_centre':'ecmwf',
        'variable':[
            '10m_wind_speed','2m_dewpoint_temperature','2m_temperature',
            'total_precipitation'
        ],
        'product_type':'monthly_mean',
        'year':[
            '2003','2004','2005',
            '2006','2007','2008',
            '2009'
        ],
        'month':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12'
        ],
        'leadtime_month':[
            '1','2','3',
            '4','5','6'
        ]
    },
    'download.grib')


for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
    c.retrieve(
        'seasonal-monthly-single-levels',
        {
            'format':'netcdf',
            'originating_centre':'ecmwf',
            'system':'5',
            'area':ind_area,
            'variable':[
                '10m_wind_speed','2m_dewpoint_temperature',
                '2m_temperature','total_precipitation', 'surface_solar_radiation_downwards'
            ],
            'product_type':[
                'monthly_mean'
            ],

            'year':[
                '1993','1994','1995',
                '1996','1997','1998',
                '1999','2000','2001',
                '2002','2003','2004',
                '2005','2006','2007',
                '2008','2009','2010',
                '2011','2012','2013',
                '2014','2015','2016', '2017', '2018'
            ],
            'month':month,
            'leadtime_month':[
                '1','2','3',
                '4','5','6'
            ]
        },
        '/mnt/data2/SEAS5/monthly/hindcasts/{0}_mean.nc'.format(month))

"""
c.retrieve(
    'seasonal-original-single-levels',
    {
        'format':'grib',
        'originating_centre':'ecmwf',
        'system':'5',
        'variable':[
            '10m_u_component_of_wind','10m_v_component_of_wind','2m_dewpoint_temperature',
            '2m_temperature','total_precipitation'
        ],
        'year':'2019',
        'month':'12',
        'day':'01',
        'area':ind_area,
        'leadtime_hour': lead_hours
    },
    '/mnt/data2/SEAS5/forecast/2019_12_ind.grib')
