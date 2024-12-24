import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import xarray as xr
#from metpy.calc import mixing_ratio_from_relative_humidity
#from metpy.units import units
import glob 

def make_monotonic(data):
    out=np.zeros(len(data))
    k=0
    for i,j in zip(data[:-1],data[1:]):
        if (i > j) or (np.isnan(j)):
            out[k] = i        
        else: 
            out[k] = np.nan
        k+=1
    mask = np.isnan(out)
    return out, mask 
    
    
def reindex_by_pressure(ds, met, time=0):
    # make a new pressure coordinate
    # new_height = np.linspace(0, 25, 100)

    # make a list of data arrays...
    list_of_variables = []

    # get the current time
    base_time = pd.to_datetime(ds.time.values)

    # get the met data fro that time 
    # met_pres = met.qcpres.sel(time = base_time, method='nearest') * 10 # convert to hPa
    # met_temp = met.qctemp.sel(time = base_time, method='nearest')
    # met_rh   = met.qcrh.sel(time = base_time, method='nearest')

    # prep the pressure data ...
    sond_pressure = ds.bar_pres.where(ds.qc_bar_pres == 0) * 10

    # add the met pressure to the bottom of the sondes 
    # combined_pressure = np.zeros(len(sond_pressure) + 1)
    # # do the same for the 
    # combined_pressure[0]  = met_pres.values 
    # combined_pressure[1:] = sond_pressure.values


    combined_pressure = np.zeros(len(sond_pressure))
    # do the same for the 
    combined_pressure[:] = sond_pressure.values


    # get the new pressure coordinate
    pressurex, mask = make_monotonic(combined_pressure)  # convert to hPa

    # do not mask out the measurement from the met station. 

    # mask out the pressure ....
    pressure = pressurex[~mask]

    # make the first entry of the mask always true... 

    # we can interpolate pressure to fewer coordinates
    reduced_pressure = np.linspace(pressure[0], 10, 50)

    ## loop through the variables 
    # for var, varmet in zip(['rh_scaled', "temp"], [met_rh, met_temp]):
    for var in ['rh_scaled', "temp"]:

        # get the data from the sondes
        data = ds[var].where(ds['qc_'+var] == 0).values

        # combine it with the met data
        # combined = np.zeros(len(data) + 1)
        # combined[0] = varmet.values
        # combined[1:] = data
        # combined = combined[~mask]

        combined = np.zeros(len(data))
        combined[:] = data
        combined = combined[~mask]

        # make a dataarray 
        da  = xr.DataArray(data=combined[:, np.newaxis].T,
                            dims=("time", "pressure"),
                            coords={"pressure":pressure,     
                                    "time": np.array([base_time])
                                })
        # give the attrs 
        da.attrs = ds[var].attrs
        da.name = ds[var].name
        list_of_variables.append(da)
    

    # give the attrs 
    da.attrs = ds[var].attrs
    da.name = ds[var].name


    # merge together these files 
    dsx = xr.merge(list_of_variables)
    dsx = dsx.drop_duplicates(dim="pressure") 
    dsx_interp = dsx.interp({"pressure":reduced_pressure}, method="linear")# kwargs={"fill_value": ""})

    dsx_interp['bar_pres'] = dsx_interp['temp'].copy()
    dsx_interp['bar_pres'].values[0,:]= dsx_interp['pressure'].values   
    dsx_interp.pressure.values[:] = np.arange(0, 100, 1)
    dsx_interp=dsx_interp.rename({"pressure":"level"})
    return dsx_interp



if __name__ == "__main__":
    
    # open up the met data 
    # met=xr.open_mfdataset("/global/homes/r/rudisill/gshare/sail_data_will/data_store_sail_period/gucmet/*.cdf")
    # met['qcpres'] = met.atmos_pressure.where(met.qc_atmos_pressure == 0)
    # met['qctemp'] = met.temp_mean.where(met.qc_temp_mean==0)
    # met['qcrh'] = met.rh_mean.where(met.qc_rh_mean==0)
    
    # this is where to put them 
    outdir='/global/homes/r/rudisill/gshare/sail_data_will/data_store_sail_period/interpsond/sond_by_pressure3/'

    # loop through them all 
    for sond in glob.glob("/global/homes/r/rudisill/gshare/sail_data_will/data_store_sail_period/interpsond/ftp.archive.arm.gov/rudisillw2/241881/gucinterpolatedsondeM1.c1.2021*"):

        # name for hte outoput data 
        outname = sond.split('/')[-1]

        # open and resample 
        sond_resample = xr.open_dataset(sond).resample(time='10min').mean()

        # get the met data for the sondes
#        ds = xr.concat([reindex_by_pressure(sond_resample.sel(time=t), met) for  t in sond_resample.time.values], dim='time')
        ds = xr.concat([reindex_by_pressure(sond_resample.sel(time=t)) for  t in sond_resample.time.values], dim='time')

        # now this will assign the "dp", which is only a function of time, to the dataset 
        dp = ds.bar_pres.diff(dim='level').isel(level=0)
        ds = ds.assign(dp=dp.drop("level"))

        # now do some filling 
        ds = ds.ffill(dim='level')
        ds = ds.bfill(dim='level')

        # write the output 
        outname = sond.split('/')[-1]
        ds.to_netcdf(outdir+outname)
        print(outname)

