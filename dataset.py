from goesaws import *
import datetime
import pandas as pd
import sys

date = str(datetime.date.today())

bucket_name = 'noaa-goes16'
product_name = 'ABI-L2-MCMIPM'
lightning_mapper = 'GLM-L2-LCFA'
yr = 2023
day_of_year =205
hr = 14
minutes = 5
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

#Generate the Lightning Data

glmprefix = gen_prefix(product = "GLM-L2-LCFA", year = yr, day = day_of_year, hour = hr)
glmfiles = gen_fn(bucket = bucket_name, prefix=glmprefix)
#There should be 180 GLM files per hour (one every 20 seconds)
glmdatasets = [gen_data(key=glmfiles[i], bucket = bucket_name) for i in range(0,minutes*3)]
latitudes = [np.array(glmdatasets[i]["event_lat"].data) for i in range(0,minutes*3)]
lats = np.concatenate(latitudes)
longitudes = [np.array(glmdatasets[i]["event_lon"].data) for i in range(0,minutes*3)]
lons= np.concatenate(longitudes)
strikes = list(zip(lats,lons))

nfile = nfile.coarsen(x=4,y=4).mean()
nfile = calc_latlon(nfile)


#make the csv file
#for i in nfile.x (where nfile has been merged and coarsened)
df_list = []
for i in range(len(nfile.x)):
    
    temp = nfile.isel(x = i)
    timebds = len(temp.t)
    lats = temp["lat"].data 
    lons = temp["lon"].data
    time = temp["t"].data
    #print(len(time))
    
    coords = list(zip(lats,lons))*timebds
    

    #y = list(temp["y"].data)*timebds
   
    dat = { 'Coordinates':coords}

    df = pd.DataFrame(dat)
    #df["Time"] = time[i]
    #df["lat"] = list(temp.lat.data)*timebds
    #df["x"] = nfile.x[i].data
    
    

    variable_list = ["CMI_C01","CMI_C02","CMI_C03","CMI_C04","CMI_C05","CMI_C06","CMI_C07","CMI_C08","CMI_C09","CMI_C10","CMI_C11","CMI_C12","CMI_C13","CMI_C14","CMI_C15", "CMI_C16" ,"ACM","BCM","Cloud_Probabilities"]

    for var in range(len(variable_list)):
        a = variable_list[var]
        df[a] = np.array([temp[a][i].data for i in range(timebds)]).flatten()
    df_list.append(df)

combined = pd.concat(df_list,axis=0)
combined = combined.reset_index()

times = nfile["t"].data
num = len(df)//minutes


time_arr = []
for time in times:
    narr = [time]*num
    time_arr.append(narr)

time_arr = [x for i in time_arr for x in i]*len(nfile.x)

combined["Time"] = time_arr

print("Made Initial Dataframe!") 

shared_coords = []

for i in range(len(strikes)):
    strike = strikes[i]
    distances = haversine_vector([strike]*len(df["Coordinates"]), df["Coordinates"].to_list())
    min_dist = min(distances)

    if min_dist < 2:
        distances = list(distances)
        idx = distances.index(min_dist)
        #assert len(distances) == len(df["Coordinates"].to_list())
        coords = df["Coordinates"].to_list()[idx]
        #assert hs.haversine(strike,coords)<5
        shared_coords.append(coords)

    
combined["Lightning"] = combined["Coordinates"].isin(shared_coords)
nmap = {True:1, False:0}
combined["Lightning"] = combined["Lightning"].map(nmap)

combined.to_csv("/Users/robbiefeldstein/Documents/Programming/Research/Datasets/" + date + ".csv")
#print(combined["Lightning"])
print(combined["Lightning"].value_counts())
