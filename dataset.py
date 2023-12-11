from goesaws import *
import datetime
from datetime import datetime
import pandas as pd
import sys
import json
from shapely.geometry import Polygon, Point

bucket_name = 'noaa-goes16'
product_name = 'ABI-L2-MCMIPM'
lightning_mapper = 'GLM-L2-LCFA'

yr = 2022
day = 21
month = 5
hr = 17
minutes = 60

date = datetime(yr,month,day,hr)
day_of_year = date.timetuple().tm_yday
#Need to be between 0 and 60


fn = str(date.date())
#Generate the ABI Datafile 

abiprefix = gen_prefix(product=product_name,year = yr, day=day_of_year, hour = hr)
abifiles = gen_fn(bucket=bucket_name,prefix=abiprefix)
abidatasets = [gen_data(key=abifiles[i], bucket = bucket_name) for i in range(0,minutes)]
abiDS = xr.concat(abidatasets,dim = 't')
#abiDS = calc_latlon(abiDS)

#Generate the Cloud Data 

skyprefix = gen_prefix(product = "ABI-L2-ACMM", year = yr, day = day_of_year, hour = hr)
skyfiles = gen_fn(bucket = bucket_name, prefix=skyprefix)
skydatasets = [gen_data(key=skyfiles[i], bucket = bucket_name) for i in range(0,minutes)]
skyDS = xr.concat(skydatasets,dim = 't')
#skyDS = calc_latlon(skyDS)

dataset2_resampled = skyDS.interp(
    x=abiDS["x"].data,
    y=abiDS["y"].data,
    t = abiDS["t"].data,
    method='linear',
    kwargs={'fill_value': "extrapolate"}
)

nfile = xr.merge([abiDS,dataset2_resampled])
nfile = nfile.coarsen(x=4,y=4).mean()
nfile = calc_latlon(nfile)

lats = nfile['lat'].values.flatten()
lons = nfile['lon'].values.flatten()

polygon = Polygon(zip(lons, lats))

# Extract the boundary coordinates as a list of tuples
boundary_coords = list(polygon.exterior.coords)
boundary_poly = Polygon(boundary_coords)

df = nfile.to_dataframe()

features = ["CMI_C01", "CMI_C02","CMI_C03","CMI_C04","CMI_C05","CMI_C06","CMI_C07","CMI_C08","CMI_C09","CMI_C10","CMI_C11","CMI_C12","CMI_C13","CMI_C14","CMI_C15","CMI_C16","ACM", "BCM", "Cloud_Probabilities","lat","lon"]
ndf = df[features].copy()
ndf["time"] = ndf.index.get_level_values('t')
ndf.drop_duplicates(inplace=True)

names = ["group_lat","group_lon","group_time_offset"]

glmprefix = gen_prefix(product = "GLM-L2-LCFA", year = yr, day = day_of_year, hour = hr)
glmfiles = gen_fn(bucket = bucket_name, prefix=glmprefix)
#There should be 180 GLM files per hour (one every 20 seconds)
glmdatasets = [gen_data(key=glmfiles[i], bucket = bucket_name) for i in range(0,minutes*3)]

latitudes = [np.array(glmdatasets[i][names[0]].data) for i in range(0,minutes*3)]
lats = np.concatenate(latitudes)

longitudes = [np.array(glmdatasets[i][names[1]].data) for i in range(0,minutes*3)]
lons= np.concatenate(longitudes)

times = [np.array(glmdatasets[i][names[2]].data) for i in range(0,minutes*3)]
times = np.concatenate(times)


points = [Point(lons[i],lats[i]) for i in range(0,len(lons))] 
is_inside = [boundary_poly.contains(points[i]) for i in range(0,len(points))]

# Remove the points that are not inside the boundary
lats = lats[is_inside]
lons = lons[is_inside]
times = times[is_inside]

strikes = list(zip(lats,lons))

from scipy.spatial import cKDTree
import pandas as pd

ndf = ndf.reset_index(drop=True)  # reset the index of the ndf dataframe
#ndf['time'] = ndf['time'].astype(np.int64)  # convert the time column to datetime objects
tree = cKDTree(ndf[['lat', 'lon']].values)
distances, indices = tree.query(strikes)

#event_distances, event_indices = tree.query(event_stikes)

# filter out the indices that are not present in the ndf dataframe
valid_indices = indices[indices < len(ndf)]

lightning_df = pd.DataFrame({
    'strike_lat': [strike[0] for strike in strikes],
    'strike_lon': [strike[1] for strike in strikes],

    # 'event_lat': [event_strike[0] for event_strike in event_stikes],
    # 'event_lon': [event_strike[1] for event_strike in event_stikes],

    'time': times, 
    #'event_time': event_times,
    'nearest_lat': ndf.loc[valid_indices, 'lat'].values,
    'nearest_lon': ndf.loc[valid_indices, 'lon'].values,

    # 'nearest_event_lat': ndf.loc[event_indices, 'lat'].values,
    # 'nearest_event_lon': ndf.loc[event_indices, 'lon'].values,


    'distance': distances[indices < len(ndf)],
    # 'event_distance': event_distances[event_indices < len(ndf)]
})


lightning_df["lightning"] = 0
lightning_df.loc[distances < 1, 'lightning'] = 1
lightning_df["Coordinates"] = list(zip(lightning_df["nearest_lat"],lightning_df["nearest_lon"]))

ndf['time_int'] = ndf['time'].astype(np.int64)

tree = cKDTree(ndf[['time_int']].values.reshape(-1, 1))

time_query = lightning_df['time'].astype(np.int64).values

distances, indices = tree.query(time_query.reshape(-1, 1))

lightning_df['nearest_time'] = ndf.loc[indices, 'time'].values

features = ["CMI_C01", "CMI_C02","CMI_C03","CMI_C04","CMI_C05","CMI_C06","CMI_C07","CMI_C08","CMI_C09","CMI_C10","CMI_C11","CMI_C12","CMI_C13","CMI_C14","CMI_C15","CMI_C16","ACM", "BCM", "Cloud_Probabilities","lat","lon","Coordinates","time","Lightning"]
ndf["Coordinates"] = list(zip(ndf["lat"],ndf["lon"]))

strike_df = lightning_df[lightning_df["lightning"] == 1]

merged_df = strike_df.merge(ndf, left_on=['nearest_time','Coordinates'], right_on=['time','Coordinates'], how='left')
#merged_df = merged_df.fillna(0)
merged_df.drop_duplicates(inplace=True)
we_want = merged_df.groupby(['Coordinates','nearest_time']).count()["lightning"].reset_index()

final_df = we_want.merge(ndf, left_on = ['nearest_time','Coordinates'], right_on = ['time','Coordinates'], how = 'right')
final_df = final_df.fillna(0)
final_df["Lightning"] = final_df["lightning"].apply(lambda x: 1 if x > 0 else 0)

final_df.to_csv("/Users/robbiefeldstein/Documents/Programming/Research/Datasets/" + fn + ".csv")
print(final_df.head())
print(fn)
print("Success!")
