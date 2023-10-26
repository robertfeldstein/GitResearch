"""
A standard Python library for importing GOES Satellite Weather Data via the cloud.
This library takes advantage of NOAA's BIG Data Project: (https://docs.opendata.aws/noaa-goes16/cics-readme.html)
All of this data can be manually combed from NASA directly, but this system is much faster. 

Mapping data
https://unidata.github.io/python-training/gallery/mapping_goes16_truecolor/

Projection Data:
https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm
"""

import xarray as xr
import requests
import netCDF4
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.pyplot as plt
import metpy
import numpy as np
import cartopy.crs as ccrs
from datetime import datetime
import pandas as pd
import seaborn as sb
import haversine as hs 
from haversine import haversine_vector
import geopy.distance
from scipy.spatial import distance
import gif

bucket_name = 'noaa-goes16'
product_name = 'ABI-L2-MCMIPM'
lightning_mapper = 'GLM-L2-LCFA'
yr = 2023
#Instead of 219
day_of_year =263
hr = 18

"""
Generates the initial file path for the AWS server.
It includes everything except for the file name. 

Requirements
hour: should be of type int and formatted in military time (0-23)
day: should be of type int and be a value between 1 and 365
product: should match a product id given at the AWS cloud documentation
"""
def gen_prefix(product=product_name, year = yr, day = day_of_year, hour = hr):
    return f'{product}/{year}/{day:03.0f}/{hour:02.0f}/'



s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
"""
Generates a list of filenames based on the initial prefix. 
"""

def gen_fn(bucket = 'noaa-goes16',client = s3_client,prefix = gen_prefix()) :
    kwargs = {'Bucket':bucket_name, 'Prefix':prefix}
    key_list = s3_client.list_objects_v2(**kwargs)["Contents"]
    keys = [item["Key"] for item in key_list]
    return keys

"""
Returns xarray dataset based on the given file name (key)

key: filename 
bucket: data bucket
"""

def gen_data(key, bucket = 'noaa-goes16'):
    resp = requests.get(f'https://{bucket}.s3.amazonaws.com/{key}')
    #print(f'https://{bucket_name}.s3.amazonaws.com/{key}')
    #Create a NetCDF File and load it using xarray into Python
    file_name = key.split('/')[-1].split('.')[0]
    nc4_ds = netCDF4.Dataset(file_name, memory = resp.content)
    store = xr.backends.NetCDF4DataStore(nc4_ds)
    DS = xr.open_dataset(store)
    return DS





# #There should be 120 ABI Datasets
# filesets = gen_fn()
# print("ABI Datasets: " ,len(filesets))
# #Pick the first mesoscale file
# file = gen_data(filesets[60])

# glmprefix = gen_prefix("GLM-L2-LCFA")
# glmfile = gen_fn(prefix=glmprefix)
# #There should be 180 GLM files per hour (one every 20 seconds)
# print("Lightning Datasets: ",len(glmfile))
# #Pick the first GLM file
# DS = gen_data(key=glmfile[0])

# #Merge the GLM data with the ABI Data
# nfile = xr.merge([file,DS])

# #Where do we see if a square has a lightning flash?
# #Want to make the data into gridded squares 

# #Upsampling to a 125x125 grid taking the mean of 4x4 grids
# #If we wanted to do the max like Prof Guinness Suggested
# nfile = nfile.coarsen(x=4,y=4).max()
# #If you just watnted to take the mean
# #nfile = nfile.coarsen(x=4,y=4).mean()



"""
Returns a pandas dataframe for one specific x coordinate given an xarrary dataset.

Requirements:

dataset: xarray dataset formed from a netcdf file from the GOES-R satellite. The data should be mesoscale to avoid slow processing. 

i: An integer value that is between 0 and len(x_values)-1, where x_values is just the list of all the x_coordinates from the ABI data.
"""
def make_csv_helper(dataset,i):
    channels = ["CMI_C01","CMI_C02","CMI_C03","CMI_C04","CMI_C05","CMI_C06","CMI_C07","CMI_C08","CMI_C09","CMI_C10","CMI_C11","CMI_C12",
                "CMI_C13","CMI_C14","CMI_C15","CMI_C16","ACM","BCM","Cloud_Probabilities","DQF"]
    column_order = ["X Data", "Y Data", "CMI_C01","CMI_C02","CMI_C03","CMI_C04","CMI_C05","CMI_C06","CMI_C07","CMI_C08","CMI_C09","CMI_C10","CMI_C11","CMI_C12",
                "CMI_C13","CMI_C14","CMI_C15","CMI_C16","ACM","BCM","Cloud_Probabilities","DQF"]
    #Grab list of the x values
    x_values = dataset.x.data
    #Grab list of the y values 
    y_values = dataset.y.data
    #Grab the actual values
    value_list = []
    
    for channel in channels:
        #print(channel)
        #isel function will return all of the y data from a given x value
        values = dataset.isel(x=i)[channel][:].data
        value_list.append(values)
    ndict = dict(zip(channels,value_list))
    df = pd.DataFrame(ndict)
    df["X Data"] = x_values[i]
    df["Y Data"] = y_values
    df = df[column_order]
    return df 

"""
Given an Xarray dataset, returns a pandas dataframe of the different ABI channels. 
"""

def make_csv(dataset):
    x_values = dataset.x.data
    df_list = []
    for i in range(len(x_values)):
        df_list.append(make_csv_helper(dataset,i))


    combined = pd.concat(df_list,axis=0)
    combined = combined.reset_index()
    return combined

# df = make_csv(nfile)


# corr = df.corr()
# sb.heatmap(corr, cmap="Blues", annot=True)
# plt.show()


"""
Next Steps:

GLM data is every 20 seconds. How do we average this into a minute? Then we would have a direct one to one mapping. 

We need to figure out how to convert the X and Y data to lat lon so that way we can know if a lightning strike occurs in a specific grid
cell or not. 

Logistic regression!

"""


"""
Note: 

There is no way to concatenate GLM data with GLM data, none of the dimensions will match because it is point series data and not measured on a time scale. 
We could simply take half of the ABI data, and 1/3 of the GLM data and merge those all into 60 netcdf files over the hour...? 

Alternatively:

There are 12 CONUS ABI Measurements taken in an hour (one every five minutes). 
Use all 12 with 12 of the GLM Data?


"""


#Plotting Imagery 
#I DID NOT WRITE THE COLOR CORRECTIONS FOR THIS
#ALL CREDIT TO: Brian Baylock
#https://unidata.github.io/python-training/gallery/mapping_goes16_truecolor/



@gif.frame
def plot_abiglm(merged):
    R = merged['CMI_C02'][:].data
    G = merged['CMI_C03'][:].data
    B = merged['CMI_C01'][:].data

    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)

    # Apply the gamma correction
    gamma = 2.2
    R = np.power(R, 1/gamma)
    G = np.power(G, 1/gamma)
    B = np.power(B, 1/gamma)

    # Calculate the "True" Green
    G_true = 0.45 * R + 0.1 * G + 0.45 * B
    G_true = np.clip(G_true, 0, 1)

    # The final RGB array :)
    RGB = np.dstack([R, G_true, B])

    dat = merged.metpy.parse_cf('CMI_C02')

    x = dat.x
    y = dat.y

    geos = dat.metpy.cartopy_crs
    

    lc = ccrs.LambertConformal(central_longitude=-97.5,
                            standard_parallels=(38.5, 38.5))

    fig = plt.figure(figsize=(10, 8))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125, -70, 25, 50], crs=ccrs.PlateCarree())
    

# Create axis with Geostationary projection
    # ax = fig.add_subplot(1, 1, 1, projection=geos)

    ax.imshow(RGB, origin='upper',
            extent=(x.min(), x.max(), y.min(), y.max()),
            transform=geos)

    #Add borders to the map
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
    ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5)
    scan_start = datetime.strptime(merged.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')

    #Plot the GLM Data
    ax.scatter(merged.event_lon[:], merged.event_lat[:],  marker = ".",color = 'blue',s=1)
    #ax.scatter(merged.group_lon[:], merged.group_lat[:],  marker = ".",color = 'green',s=25)
    #ax.scatter(merged.flash_lon[:], merged.flash_lat[:],  marker = ".",color = 'red',s=1)

    plt.title('GOES-16 True Color', fontweight='bold', fontsize=15, loc='left')
    plt.title('CONUS 1')
    plt.title('{}'.format(scan_start.strftime('%H:%M UTC %d %B %Y')), loc='right')
    #plt.show()




def calc_latlon(ds):
    # The math for this function was taken from 
    # https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm
    x = ds.x
    y = ds.y
    goes_imager_projection = ds.goes_imager_projection
    
    x,y = np.meshgrid(x,y)
    
    r_eq = goes_imager_projection.attrs["semi_major_axis"]
    r_pol = goes_imager_projection.attrs["semi_minor_axis"]
    l_0 = goes_imager_projection.attrs["longitude_of_projection_origin"] * (np.pi/180)
    h_sat = goes_imager_projection.attrs["perspective_point_height"]
    H = r_eq + h_sat
    
    a = np.sin(x)**2 + (np.cos(x)**2 * (np.cos(y)**2 + (r_eq**2 / r_pol**2) * np.sin(y)**2))
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - r_eq**2
    
    r_s = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    
    s_x = r_s * np.cos(x) * np.cos(y)
    s_y = -r_s * np.sin(x)
    s_z = r_s * np.cos(x) * np.sin(y)
    
    lat = np.arctan((r_eq**2 / r_pol**2) * (s_z / np.sqrt((H-s_x)**2 +s_y**2))) * (180/np.pi)
    lon = (l_0 - np.arctan(s_y / (H-s_x))) * (180/np.pi)
    
    ds = ds.assign_coords({
        "lat":(["y","x"],lat),
        "lon":(["y","x"],lon)
    })
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"
    return ds

def find_distance(pt, coord):
    return hs.haversine(pt,coord)


def one_minute_file(abi_file_idx, minute):
    print("Starting File #: ", abi_file_idx+ 1)
    #ABI DATA
    filesets = gen_fn()
    file = gen_data(filesets[abi_file_idx])

    skymask = gen_prefix("ABI-L2-ACMM")
    skyfile = gen_fn(prefix=skymask)
    #print("Sky Datasets: ", len(skyfile))
    skyDS = gen_data(key=skyfile[abi_file_idx])

#Merge the GLM data with the ABI Data
    nfile = xr.merge([file,skyDS],compat='override')
    #print(list(nfile.keys()))
    nfile = nfile.coarsen(x=4,y=4).mean()

    

    df = make_csv(nfile)
    latlonfile = calc_latlon(nfile)
    guess_lats = latlonfile.lat.data.flatten()
    guess_lons = latlonfile.lon.data.flatten()
    df["Lat"] = guess_lats
    df["Lon"] = guess_lons
    df["Coordinates"] = list(zip(df["Lat"],df["Lon"]))
    df["Time"] = file.t.data


    #Lightning Data
    glmprefix = gen_prefix("GLM-L2-LCFA")
    glmfile = gen_fn(prefix=glmprefix)
    
    lightning_indices = [2*minute,2*minute+1,2*minute+2]
    DS_1 = gen_data(key=glmfile[lightning_indices[0]])
    DS_2 = gen_data(key=glmfile[lightning_indices[1]])
    DS_3 = gen_data(key=glmfile[lightning_indices[2]])
    xarraybucket = [DS_1,DS_2,DS_3]

    #generate all of the lat lon pairs of the lightning data
    lats = [list(DS["flash_lat"].data.flatten()) for DS in xarraybucket]
    lons = [list(DS["flash_lon"].data.flatten()) for DS in xarraybucket]

    lat_coords = []
    for arr in lats:
        for val in arr:
            lat_coords.append(val)

    lon_coords = []
    for arr in lons:
        for val in arr:
            lon_coords.append(val)

    lats = lat_coords
    lons = lon_coords

    #Where all the lightning struck
    lightningstrikes = list(zip(lats,lons))

    #We need to find something that works faster...

    #Closest coordinates to the ABI data
    shared_coords = []
    for i in range(len(lightningstrikes)):
        strike = lightningstrikes[i]
        distances = haversine_vector([strike]*len(df["Coordinates"]), df["Coordinates"].to_list())
        min_dist = min(distances)

        if min_dist < 2:
            distances = list(distances)
            idx = distances.index(min_dist)
            #assert len(distances) == len(df["Coordinates"].to_list())
           
            coords = df["Coordinates"].to_list()[idx]
            #assert hs.haversine(strike,coords)<5
            shared_coords.append(coords)

    
    df["Lightning"] = df["Coordinates"].isin(shared_coords)
    nmap = {True:1, False:0}
    df["Lightning"] = df["Lightning"].map(nmap)
    
    return df


#print(one_minute_file(60,0))

# filesets = gen_fn()
# print("ABI Datasets: " ,len(filesets))
# #Pick the first mesoscale file
# file = gen_data(filesets[60])

# glmprefix = gen_prefix("GLM-L2-LCFA")
# glmfile = gen_fn(prefix=glmprefix)
# #There should be 180 GLM files per hour (one every 20 seconds)
# print("Lightning Datasets: ",len(glmfile))
# #Pick the first GLM file
# DS = gen_data(key=glmfile[0])

# #Merge the GLM data with the ABI Data
# nfile = xr.merge([file,DS])

# #Where do we see if a square has a lightning flash?
# #Want to make the data into gridded squares 

# #Upsampling to a 125x125 grid taking the mean of 4x4 grids
# #If we wanted to do the max like Prof Guinness Suggested
# nfile = nfile.coarsen(x=4,y=4).max()
# plot_abiglm(nfile)

#Find the nearest pixel to every lightning flash
#Assign this 

