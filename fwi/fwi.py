import numpy as np
import pandas as pd
import xarray as xr

class FWICalc_pixel:
    def __init__(self, temp, rhum, wind, prcp):
        self.rhum = rhum
        self.temp = temp
        self.wind = wind
        self.prcp = prcp

    def FFMCcalc(self, ffmc0):
        mo = (147.2 * (101.0 - ffmc0)) / (59.5 + ffmc0) #*Eq. 1*#
        if self.prcp > 0.5:
            rf = self.prcp - 0.5                       #*Eq. 2*#
            if mo > 150.0:
                mo = (mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) *
                    (1.0 - np.exp(-6.93 / rf))) \
                    + (0.0015 * (mo - 150.0)**2) * np.sqrt(rf) #*Eq. 3b*# 
            else:
                mo = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 -
                np.exp(-6.93 / rf))                   #*Eq. 3a*#
        if mo > 250.0:
            mo = 250.0
        ed = 0.942 * self.rhum**0.679 + (11.0 * np.exp((self.rhum - 100.0) / 10.0)) \
            + 0.18 * (21.1 - self.temp) * (1.0 - 1.0 / np.exp(0.1150 * self.rhum)) #*Eq. 4*#

        if mo < ed:
            ew = 0.618 * self.rhum**.753 + (10.0 * np.exp((self.rhum - 100.0) / 10.0)) \
                + 0.18 * (21.1 - self.temp) * (1.0 - 1.0 / np.exp(0.115 * self.rhum))         #*Eq. 5*#
            if mo <= ew:
                kl = 0.424 * (1.0 - ((100.0 - self.rhum) / 100.0)**1.7) + (0.0694 * np.sqrt(self.wind)) \
                    * (1.0 - ((100.0 - self.rhum) / 100.0)**8)       #*Eq. 7a*#
                kw = kl * (0.581 * np.exp(0.0365 * self.temp))                 #*Eq. 7b*#
                mm = ew - (ew - mo) / 10.0**kw                                  #*Eq. 9*#
            elif mo > ew:
                mm = mo
        elif mo == ed:
            mm = mo
        elif mo > ed:
            kl = 0.424 * (1.0 - (self.rhum / 100.0)**1.7) + (0.0694 * np.sqrt(self.wind)) \
                * (1.0 - (self.rhum / 100.0)**8)                    #*Eq. 6a*#
            kw = kl * (0.581 * np.exp(0.0365 * self.temp))                        #*Eq. 6b*#
            mm  = ed + (mo - ed) / 10.0**kw                                    #*Eq. 8*#
        ffmc = (59.5 * (250.0 - mm)) / (147.2 + mm)
        if ffmc  > 101.0:
            ffmc = 101.0
        if ffmc  <= 0.0:
            ffmc = 0.0
        return ffmc

    def DMCcalc(self,dmc0,mth):
        el = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
        if self.temp < -1.1:
            self.temp = -1.1
        rk = 1.894 * (self.temp + 1.1) * (100.0 - self.rhum) * (el[mth - 1] * 0.0001)    #*Eqs. 16 and 17*#
        if self.prcp > 1.5:
            rw = 0.92 * self.prcp - 1.27
            wmi = 20.0 + 280.0 / np.exp(0.023 * dmc0)
            if dmc0 <= 33.0:
                b = 100.0 / (0.5 + 0.3 * dmc0)
            elif dmc0 > 33.0:
                if dmc0 <= 65.0:
                    b = 14.0 - 1.3 * np.log(dmc0)
                elif dmc0 > 65.0:
                    b = 6.2 * np.log(dmc0) - 17.2
            wmr = wmi + (1000 * rw) / (48.77 + b * rw)
            pr = 43.43 * (5.6348 - np.log(wmr - 20.0))
        elif self.prcp <= 1.5:
            pr = dmc0
        if pr < 0.0:
            pr = 0.0
        dmc = pr + rk
        if dmc <= 1.0:
            dmc = 1.0
        return dmc #*Eq. 11*##*Eq. 12*##*Eq. 13a*##*Eq. 13b*##*Eq. 13c*##*Eq. 14*##*Eq. 15*#

    def DCcalc(self, dc0, mth):
        fl = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
        if self.temp < -2.8:
            self.temp = -2.8
        pe = (0.36 * (self.temp + 2.8) + fl[mth - 1]) / 2
        if pe <= 0.0:
            pe = 0.0 #*Eq. 22*#
        if self.prcp > 2.8:
            rw = 0.83 * self.prcp - 1.27 #*Eq. 18*# 
            smi = 800.0 * np.exp(-dc0 / 400.0)                              #*Eq. 19*#
            dr = dc0 - 400.0 * np.log(1.0 + ((3.937 * rw) / smi))      #*Eqs. 20 and 21*#
            if dr > 0.0:
                dc = dr + pe
        elif self.prcp <= 2.8:
            dc = dc0 + pe
        return dc

    def ISIcalc(self, ffmc):
        mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        ff = 19.115 * np.exp(mo * -0.1386) * (1.0 + mo**5.31 / 49300000.0)
        isi = ff * np.exp(0.05039 * self.wind)
        return isi

    def BUIcalc(self,dmc,dc):
        if dmc <= (0.4 * dc):
            bui = (0.8 * dc * dmc) / (dmc + 0.4 * dc)
        else:
            bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc)**1.7)
        if bui < 0.0:
            bui = 0.0
        return bui

    def FWIcalc(self, isi, bui):
        if bui <= 80.0:
            bb = 0.1 * isi * (0.626 * bui**0.809 + 2.0)
        else:
            bb = 0.1 * isi * (1000.0 / (25. + 108.64 / np.exp(0.023 * bui)))
        if bb <= 1.0:
            fwi = bb
        else:
            fwi = np.exp(2.72 * (0.434 * np.log(bb))**0.647)
        return fwi

if __name__ == '__main__':
    #testing
    ffmc0 = 85.0
    dmc0 = 6.0
    dc0 = 15.0
    ds = pd.read_csv('test_fwi.csv', sep = ',')
    ds = ds.iloc[:4, :]
    for name, row in ds.iterrows():
        mth, day, temp, rhum, wind, prcp = row[['Month', 'Day', 'Temp.', 'RH', 'Wind', 'Rain']]
        if rhum > 100.0:
            rhum = 100.0
        mth = int(mth)
        fwisystem = FWICalc_pixel(temp, rhum, wind, prcp)
        ffmc = fwisystem.FFMCcalc(ffmc0)
        dmc = fwisystem.DMCcalc(dmc0, mth)
        dc = fwisystem.DCcalc(dc0, mth)
        isi = fwisystem.ISIcalc(ffmc)
        bui = fwisystem.BUIcalc(dmc, dc)
        fwi = fwisystem.FWIcalc(isi, bui)
        print('{0:.1f} {1:.1f} {2:.1f} {3:.1f} {4:.1f} {5:.1f}'.format(ffmc, dmc, dc, isi, bui, fwi))
        ffmc0 = ffmc
        dmc0 = dmc
        dc0 = dc


