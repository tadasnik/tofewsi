import numpy as np
import pandas as pd
import xarray as xr
#from fwi import FWICalc_pixel

def pixel_wise(fwi_sel):
    ffmc0 = 85.0
    dmc0 = 6.0
    dc0 = 15.0
    arr_shape = [fwi_sel.dims[x] for x in ['latitude', 'longitude']]
    results = []
    mth, day = fwi_sel['time.month'].values, fwi_sel['time.day'].values
    for item in zip(fwi_sel['t2m'].values.flatten(),
                    fwi_sel['h2m'].values.flatten(),
                    fwi_sel['w10'].values.flatten(),
                    fwi_sel['tp'].values.flatten()):
        temp = item[0]
        rhum = item[1]
        wind = item[2]
        prcp = item[3]
        mth = int(mth)
        fs= FWICalc_pixel(temp, rhum, wind, prcp)
        ffmc = fs.FFMCcalc(ffmc0)
        dmc = fs.DMCcalc(dmc0, mth)
        dc = fs.DCcalc(dc0, mth)
        isi = fs.ISIcalc(ffmc)
        bui = fs.BUIcalc(dmc, dc)
        fwi = fs.FWIcalc(isi, bui)
        results.append(fwi)
    return np.array(results).reshape(arr_shape)

def compare_vector_pixel(temp, rhum, wind, prcp, mth, day):
    arr_shape = (5,5)
    print(temp, rhum, wind, prcp)
    ffmc0 = 85.0
    dmc0 = 6.0
    dc0 = 15.0
    mth = int(mth)
    fs= FWICalc_pixel(temp, rhum, wind, prcp)
    ffmc = fs.FFMCcalc(ffmc0)
    dmc = fs.DMCcalc(dmc0, mth)
    dc = fs.DCcalc(dc0, mth)
    isi = fs.ISIcalc(ffmc)
    bui = fs.BUIcalc(dmc, dc)
    fwi = fs.FWIcalc(isi, bui)
    print('pixel', ffmc, dmc, dc, fwi)
    ffmc0 = np.full(arr_shape, 85.0)
    dmc0 = np.full(arr_shape, 6.0)
    dc0 = np.full(arr_shape, 15.0)
    fs = FWICalc(np.full(arr_shape,temp), np.full(arr_shape,rhum),
                        np.full(arr_shape, wind), np.full(arr_shape,prcp))
    ffmc = fs.FFMCcalc(ffmc0)
    dmc = fs.DMCcalc(dmc0, mth)
    dc = fs.DCcalc(dc0, mth)
    isi = fs.ISIcalc(ffmc)
    bui = fs.BUIcalc(dmc, dc)
    fwi = fs.FWIcalc(isi, bui)
    print('vect', ffmc[0][0], dmc[0][0], dc[0][0], fwi[0][0])
    #print('{0:.1f} {1:.1f} {2:.1f} {3:.1f} {4:.1f} {5:.1f}'.format(ffmc, dmc, dc, isi, bui, fwi))

class FWI:
    def __init__(self, temp, rhum, wind, prcp):
        self.rhum = rhum
        self.temp = temp
        self.wind = wind
        self.prcp = prcp

    def FFMCcalc(self, ffmc0):
        # make intermediate coefficient arrays
        kl = np.ones_like(ffmc0)
        kw = np.ones_like(ffmc0)
        mm = np.ones_like(ffmc0)

        mo = (147.2 * (101.0 - ffmc0)) / (59.5 + ffmc0) #*Eq. 1*#
        if self.prcp.max() > 0.5:
            prcp_mask = self.prcp > 0.5
            rf = np.ones_like(ffmc0)
            rf[prcp_mask] = self.prcp[prcp_mask] - 0.5                       #*Eq. 2*#
            if mo[prcp_mask].max() > 150.0:
                mo_mask = prcp_mask & (mo > 150.0)
                mo[mo_mask] = ((mo[mo_mask] + 42.5 * rf[mo_mask] *
                              np.exp(-100.0 / (251.0 - mo[mo_mask])) *
                              (1.0 - np.exp(-6.93 / rf[mo_mask]))) +
                              (0.0015 * (mo[mo_mask] - 150.0)**2) *
                              np.sqrt(rf[mo_mask])) #*Eq. 3b*# 
            else:
                mo_mask = prcp_mask & (mo < 150.0)
                mo[mo_mask] = (mo[mo_mask] + 42.5 * rf[mo_mask] *
                                np.exp(-100.0 / (251.0 - mo[mo_mask])) *
                                (1.0 - np.exp(-6.93 / rf[mo_mask])))        #*Eq. 3a*#
        mo[mo > 250.0] = 250.0
        ed = 0.942 * self.rhum**0.679 + (11.0 * np.exp((self.rhum - 100.0) / 10.0)) \
            + 0.18 * (21.1 - self.temp) * (1.0 - 1.0 / np.exp(0.1150 * self.rhum)) #*Eq. 4*#

        if np.any(mo < ed):
            mo_less = mo < ed
            ew = 0.618 * self.rhum**.753 + (10.0 * np.exp((self.rhum - 100.0) / 10.0)) \
                + 0.18 * (21.1 - self.temp) * (1.0 - 1.0 / np.exp(0.115 * self.rhum))         #*Eq. 5*#
            if np.any(mo[mo_less] <= ew[mo_less]):
                mo_ew = mo_less & (mo <= ew)
                kl[mo_ew] = (0.424 * (1.0 - ((100.0 - self.rhum[mo_ew]) / 100.0)**1.7) +
                            (0.0694 * np.sqrt(self.wind[mo_ew])) *
                            (1.0 - ((100.0 - self.rhum[mo_ew]) / 100.0)**8))      #*Eq. 7a*#
                kw[mo_ew] = kl[mo_ew] * (0.581 * np.exp(0.0365 * self.temp[mo_ew]))                 #*Eq. 7b*#
                mm[mo_ew] = ew[mo_ew] - (ew[mo_ew] - mo[mo_ew]) / 10.0**kw[mo_ew]                                  #*Eq. 9*#
            if np.any(mo[mo_less] > ew[mo_less]):
                mo_ew = mo_less & (mo > ew)
                mm[mo_ew] = mo[mo_ew]
        if np.any(mo == ed):
            mo_eq = mo == ed
            mm[mo_eq] = mo[mo_eq]

        if np.any(mo > ed):
            mo_more = mo > ed
            kl[mo_more] = (0.424 * (1.0 - (self.rhum[mo_more] / 100.0)**1.7) +
                 (0.0694 * np.sqrt(self.wind[mo_more])) *
                 (1.0 - (self.rhum[mo_more] / 100.0)**8))                    #*Eq. 6a*#
            kw[mo_more] = kl[mo_more] * (0.581 * np.exp(0.0365 * self.temp[mo_more]))                        #*Eq. 6b*#
            mm[mo_more]  = ed[mo_more] + (mo[mo_more] - ed[mo_more]) / 10.0**kw[mo_more]                                    #*Eq. 8*#

        ffmc = (59.5 * (250.0 - mm)) / (147.2 + mm)
        ffmc[ffmc > 101.0] = 101.0
        ffmc[ffmc <= 0.0] = 0.0
        return ffmc

    def DMCcalc(self, dmc0, mth):
        el = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
        self.temp[self.temp < -1.1] = -1.1
        rk = 1.894 * (self.temp + 1.1) * (100.0 - self.rhum) * (el[mth - 1] * 0.0001)    #*Eqs. 16 and 17*#
        pr = np.ones_like(dmc0)
        if np.any(self.prcp > 1.5):
            bb = np.ones_like(dmc0)
            prcp_mask = self.prcp > 1.5
            rw = 0.92 * self.prcp[prcp_mask] - 1.27
            wmi = 20.0 + 280.0 / np.exp(0.023 * dmc0[prcp_mask])
            if np.any(dmc0 <= 33.0):
                bb[dmc0 <= 33.0] = 100.0 / (0.5 + 0.3 * dmc0[dmc0 <= 33.0])
            if np.any((dmc0 > 33.0) & (dmc0 <= 65.0)):
                dmc0_mask = (dmc0 > 33.0) & (dmc0 <= 65.0)
                bb[dmc0_mask] = 14.0 - 1.3 * np.log(dmc0[dmc0_mask])
            if np.any(dmc0 > 65.0):
                dmc0_mask = dmc0 <= 65.0
                bb[dmc0_mask] = 6.2 * np.log(dmc0[dmc0_mask]) - 17.2
            wmr = wmi + (1000 * rw) / (48.77 + bb[prcp_mask] * rw)
            pr[prcp_mask] = 43.43 * (5.6348 - np.log(wmr - 20.0))
        if np.any(self.prcp <= 1.5):
            pr[self.prcp <= 1.5] = dmc0[self.prcp <= 1.5]
        pr[pr < 0.0] = 0.0
        dmc = pr + rk
        dmc[dmc <= 1.0] = 1.0
        return dmc

    def DCcalc(self, dc0, mth):
        fl = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
        self.temp[self.temp < -2.8] = -2.8
        pe = (0.36 * (self.temp + 2.8) + fl[mth - 1]) / 2
        pe[pe <= 0.0] = 0.0
        dc = np.zeros_like(dc0)
        dr = np.zeros_like(dc0)
        if np.any(self.prcp > 2.8):
            prcp_mask = self.prcp > 2.8
            rw = 0.83 * self.prcp[prcp_mask] - 1.27 #*Eq. 18*# 
            smi = 800.0 * np.exp(-dc0[prcp_mask] / 400.0)                              #*Eq. 19*#
            dr[prcp_mask] = dc0[prcp_mask] - 400.0 * np.log(1.0 + ((3.937 * rw) / smi))      #*Eqs. 20 and 21*#
            if np.any(dr > 0.0):
                dc[dr > 0.0] = dr[dr > 0.0] + pe[dr > 0.0]
        if np.any(self.prcp <= 2.8):
            prcp_less = self.prcp <= 2.8
            dc[prcp_less] = dc0[prcp_less] + pe[prcp_less]
        return dc

    def ISIcalc(self, ffmc):
        mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        ff = 19.115 * np.exp(mo * -0.1386) * (1.0 + mo**5.31 / 49300000.0)
        isi = ff * np.exp(0.05039 * self.wind)
        return isi

    def BUIcalc(self, dmc, dc):
        bui = np.zeros_like(dmc)
        dmc_mask = dmc <= (0.4 * dc)
        if np.any(dmc <= (0.4 * dc)):
            bui[dmc_mask] = ((0.8 * dc[dmc_mask] * dmc[dmc_mask]) /
                            (dmc[dmc_mask] + 0.4 * dc[dmc_mask]))
        if np.any(dmc > (0.4 * dc)):
            bui[~dmc_mask] = (dmc[~dmc_mask] - (1.0 - 0.8 * dc[~dmc_mask] /
                             (dmc[~dmc_mask] + 0.4 * dc[~dmc_mask])) *
                             (0.92 + (0.0114 * dmc[~dmc_mask])**1.7))
        bui[bui < 0.0] = 0.0
        return bui

    def FWIcalc(self, isi, bui):
        bb = np.zeros_like(isi)
        fwi = np.zeros_like(isi)
        if np.any(bui <= 80.0):
            bui_mask = bui <= 80.0
            bb[bui_mask] = 0.1 * isi[bui_mask] * (0.626 * bui[bui_mask]**0.809 + 2.0)
        if np.any(bui > 80.0):
            bui_mask = bui > 80.0
            bb[bui_mask] = (0.1 * isi[bui_mask] *
                            (1000.0 / (25. + 108.64 / np.exp(0.023 * bui[bui_mask]))))
        bb_mask = bb <= 1.0
        if np.any(bb <= 1.0):
            fwi[bb_mask] = bb[bb_mask]
        if np.any(bb[~bb_mask]):
            fwi[~bb_mask] = np.exp(2.72 * (0.434 * np.log(bb[~bb_mask]))**0.647)
        return fwi

if __name__ == '__main__':
    for year in [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]:
        fwi_arr = xr.open_dataset('~/tofewsi/data/rh_temp_wind_prcp_amazon_{0}.nc'.format(year))
        arr_shape = [fwi_arr.dims[x] for x in ['latitude', 'longitude']]
        lats = fwi_arr.latitude.values
        lons = fwi_arr.longitude.values
        times = fwi_arr.time


        #Arrays with initial conditions
        ffmc0 = np.full(arr_shape, 85.0)
        dmc0 = np.full(arr_shape, 6.0)
        dc0 = np.full(arr_shape, 15.0)


        dcs = []
        fwis = []
        #Iterrate over time dimension
        for tt in times:
            print(tt)
            fwi_sel = fwi_arr.sel(time = tt)
            mth, day = fwi_sel['time.month'].values, fwi_sel['time.day'].values
            fs = FWI(fwi_sel['t2m'].values,
                     fwi_sel['h2m'].values,
                     fwi_sel['w10'].values,
                     fwi_sel['tp'].values)
            ffmc = fs.FFMCcalc(ffmc0)
            dmc = fs.DMCcalc(dmc0, mth)
            dc = fs.DCcalc(dc0, mth)
            isi = fs.ISIcalc(ffmc)
            bui = fs.BUIcalc(dmc, dc)
            fwi = fs.FWIcalc(isi, bui)
            ffmc0 = ffmc
            dmc0 = dmc
            dc0 = dc
            dcs.append(dc)
            fwis.append(fwi)

        dataset = xr.Dataset({'dc': (['time', 'latitude', 'longitude'], dcs),
            'fwi': (['time', 'latitude', 'longitude'], fwis)},
            coords={'latitude': lats,
                    'longitude': lons,
                    'time': times})
        dataset.to_netcdf('../data/fwi_dc_amazon{0}.nc'.format(year))


    """
    #testing
    darrays = []
    ds = pd.read_csv('test_fwi.csv', sep = ',')
    ds = ds.iloc[:4, :]
    for name, row in ds.iterrows():
        mth, day, temp, rhum, wind, prcp = row[['Month', 'Day', 'Temp.', 'RH', 'Wind', 'Rain']]
        print(mth, day)
        if rhum > 100.0:
            rhum = 100.0
        mth = int(mth)
        fs = FWICalc(np.full(arr_shape,temp), np.full(arr_shape,rhum),
                            np.full(arr_shape, wind), np.full(arr_shape,prcp))
        ffmc = fs.FFMCcalc(ffmc0)
        dmc = fs.DMCcalc(dmc0, mth)
        dc = fs.DCcalc(dc0, mth)
        isi = fs.ISIcalc(ffmc)
        bui = fs.BUIcalc(dmc, dc)
        fwi = fs.FWIcalc(isi, bui)
        print('ffmc0',ffmc0[0][0])
        print('ffmc', ffmc[0][0])
        print('fwi', fwi[0][0])
        #print('{0:.1f} {1:.1f} {2:.1f} {3:.1f} {4:.1f} {5:.1f}'.format(ffmc, dmc, dc, isi, bui, fwi))
        ffmc0 = ffmc
        dmc0 = dmc
        dc0 = dc
    """

