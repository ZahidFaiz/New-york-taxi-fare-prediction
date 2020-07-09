#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv" , nrows = 100,low_memory=False)


# In[ ]:


df.head(10)


# Checking out the dtypes shows us that there are some string values in the data set

# In[ ]:


df.dtypes


# Checking out the null values now 

# In[ ]:


df.isnull().sum()


# There are only 33 values in 5_000_000 rows.. so we just drop them
# Also we convert those string dtypes to float in the data set

# Now we check for the outliers and other things like mean,SD and min max values for 5_000_000 * 5 rows

# In[ ]:


def calculate_describe(chunks):
    return chunks.describe().values


# In[ ]:


from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, long1, lat2, long2,df):
    data = df
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    R = 6371
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
    
    #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
    #c = 2 * atan2( √a, √(1−a) )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    #d = R*c
    d = (R * c) #in kilometers
    df['Distance'] = d
    return d


# Function to convert dtypes of pickup date time from objects to datetime format, and dropping the null values 

# In[ ]:


def data_cleaning(chunks):
    chunks.dropna(axis = 0, inplace = True)
    chunks.fare_amount = pd.to_numeric(chunks.fare_amount, errors='coerce')
    chunks.passenger_count = pd.to_numeric(chunks.passenger_count, errors='coerce')    
    chunks.pickup_latitude = pd.to_numeric(chunks.pickup_latitude, errors='coerce')
    chunks.pickup_longitude = pd.to_numeric(chunks.pickup_longitude, errors='coerce')
    chunks.dropoff_latitude = pd.to_numeric(chunks.dropoff_latitude, errors='coerce')
    chunks.dropoff_longitude = pd.to_numeric(chunks.dropoff_longitude, errors='coerce')
    return chunks


# Function to plot on map

# In[ ]:


def mapplots(df):
    import plotly.express as px
    fig = px.scatter_mapbox(df, lat="pickup_latitude", lon="pickup_longitude", hover_name="passenger_count", hover_data=['passenger_count','fare_amount'],
                        color_discrete_sequence=["black"], zoom=3, height=300)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()


# In[ ]:


def add_new_date_time_features(dataset):
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['year'] = dataset.pickup_datetime.dt.year
    dataset['day_of_week'] = dataset.pickup_datetime.dt.dayofweek
    
    return dataset


# In[ ]:


mapplots(df.head(1000))


# In[ ]:


chunks1 = 5_000
i = 0 
z = []
null1 = []
for chunks in pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv", chunksize = chunks1,low_memory=False):
    chunks =data_cleaning(chunks)
    mapplots(chunks)
    print(chunks.shape)
    print(chunks.dtypes)
    y = calculate_describe(chunks)
    i = i+1
    print (i)
    z.append(y)
    if(i==400):
        break


# As we can see there are outliers that we have to remove

# In[ ]:


count = sum([z[i][0][0] for i in range(len(z))])
fare_amount = sum(z[i][1][0] for i in range(len(z)))
pickup_longitude = sum(z[i][1][1] for i in range(len(z)))
pickup_latitude = sum(z[i][1][2] for i in range(len(z)))
dropoff_longitude = sum(z[i][1][3] for i in range(len(z)))
passenger_count = sum(z[i][1][4] for i in range(len(z)))
mean_fare_amount = fare_amount/len(z)
mean_pickup_longitude = pickup_longitude/len(z)
mean_dropoff_longitude = dropoff_longitude/len(z)
mean_passenger_count = passenger_count/len(z)


# In[ ]:


maximum_fare = max([z[i][7][0] for i in range(len(z))])
print ("maximum fare:", maximum_fare)
min_fare = min([z[i][3][0] for i in range(len(z))])
print ("min_fare",min_fare)
max_passenger = max([z[i][7][5] for i in range(len(z))])
print("max_passenger",max_passenger)
min_passenger = max([z[i][3][5] for i in range(len(z))])
print("min_passenger",min_passenger)


# In[ ]:


print("Mean fare amount:",mean_fare_amount)
print ("mean pickup latitude and longitude:",mean_pickup_longitude,mean_dropoff_longitude)
print ("Mean passenger count",mean_passenger_count)


# Checking outlieres using histograms for the fare amount and passenger count 

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv" , nrows = 200_00,low_memory=False)


# In[ ]:


df =data_cleaning(df)


# In[ ]:


df =data_cleaning(df)
df.fare_amount.hist(bins=30, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram')


# In[ ]:


df.isnull().sum()


# In[ ]:


df =data_cleaning(df)
df.passenger_count.hist(bins=10, figsize=(14,3))
plt.xlabel('Passenger count')
plt.title('Histogram')


# Plotting the data on histogram gave us some intution about how the data is distributed but not gives us clear insight about  how many outliers are there, so we plot Boxplot 

# In[ ]:


import seaborn as sns


# In[ ]:


sns.boxplot(x = df.fare_amount)


# In[ ]:


sns.boxplot(x = df.passenger_count)


# In[ ]:


sns.boxplot(x = df[df.passenger_count<12].passenger_count)


# For latitude and longitude we will use the mapplot 

# In[ ]:


import plotly.express as px

fig = px.scatter_mapbox(df, lat="pickup_latitude", lon="pickup_longitude", hover_name="passenger_count", hover_data=['passenger_count','fare_amount'],
                        color_discrete_sequence=["black"], zoom=3, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# as we can see there are many outliers in the map also, we have to drop these values as we cannot use this in out models, and also there is not much information about these latitude and longitude 

# first we need to remove the outliers 
# Outliers could be :
# 
# 1. Some values in passenger count is more then 12 which is not possible, some values are negative which is also not possible.
# 
# 2. Also some are zero this means that they could be carrying lagguage.
# 
# 3. Some values in fare count is also negative which is also physically not possible. 
# 
# 4. Some fare are zero even if the pickup location is not zero. It could mean that the customer cancelled the cab after it has reached the loacation . 
# 
# 5. And the most important outliers are the ones where the pickup and dropoff locations are wayy too outside NewYork, which may be the result of the incorrect data input 
# 
# Other then that there are some fare values which are above 400 but this could be because that could be fix price of the cabs. 
# 

# For removing the outliers we will be doing these things :
# 
# 1. Remove the passenge count above 12
# 2. Keep the zero passenger count rows as they contain the fare amount but drop the negative values
# 3. Drop the negative fare amount rows
# 4. 
# 5. And we will drop the values which are wayy out of NewYork (who takes cabs to travel to other continents)

# Taxi fares for NYC according to its government is 
# * -- $2.50 initial charge.
# 
# * -- Plus 50 cents per 1/5 mile when traveling above 12mph or per 60 seconds in slow traffic or when the vehicle is stopped.
# 
# * -- Plus 50 cents MTA State Surcharge for all trips that end in New York City or Nassau, Suffolk, Westchester, Rockland,     Dutchess, Orange or Putnam Counties.
# 
# * -- Plus 30 cents Improvement Surcharge.
# 
# * -- Plus 50 cents overnight surcharge 8pm to 6am.
# 
# * -- Plus $1.00 rush hour surcharge from 4pm to 8pm on weekdays, excluding holidays.
# 
# * -- Plus New York State Congestion Surcharge of $2.50 (Yellow Taxi) or $2.75 (Green Taxi and FHV) or 75 cents (any shared ride) for all trips that begin, end or pass through Manhattan south of 96th Street.
# * -- Plus tips and any tolls.

# 
# 
# min_lat <- 40.5774
# max_lat <- 40.9176
# min_long <- -74.15
# max_long <- -73.7004

# In[ ]:


chunks1 = 5_000_00
i = 0 
for chunks in pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv", chunksize = chunks1,low_memory=False):
    chunks =data_cleaning(chunks)
    i = i+1
    mapplots(chunks[chunks.dropoff_latitude>=41])
    if (i == 2):
        break


# Checking for the outliers in the map. Some rows doesn't have pickup or dropoff point (zero in coodinates) but still has fare amount in that row. So we cannot remove these as they are important for us. 
# 

# In[ ]:


chunks1 = 10_000_0
z = []
i = 0 
for chunks in pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv", chunksize = chunks1,low_memory=False):
    chunks =data_cleaning(chunks)
    chunks
    i = i+1
    chunks=chunks[(chunks.pickup_latitude>40.914)|(chunks.pickup_longitude>-73.6948)|(chunks.pickup_latitude<40.49301)|(chunks.pickup_longitude>-73.6931)|(chunks.pickup_longitude<-74.5267)]
    z.append(chunks.shape)
    if (i == 500):
        break


# In[ ]:


#number of coordinates outside new york
i = 0
out_nyc = sum(z[i][0] for i in range(len(z)))
print("Number of coordinates outside new york:",out_nyc)


# Dropping these rows using the code below

# In[ ]:


def map_box(dft):
    dft.drop(index=dft[(dft.pickup_longitude <= -75) 
                         | (dft.pickup_longitude >= -72) 
                         | (dft.dropoff_longitude <= -75) 
                         | (dft.dropoff_longitude >= -72)
                         | (dft.pickup_latitude <= 39)
                         | (dft.pickup_latitude >= 42)
                         | (dft.dropoff_latitude <= 39)
                         | (dft.dropoff_latitude >= 42)].index, inplace=True)
    return dft


# Calculating distance between two points, adding this feature as this directly effects the fare amount

# In[ ]:


haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',df)
df.head(10)


# Removing the negative values in the fare column 

# In[ ]:


print("old size: %d" % len(df))
df = df[df.fare_amount >=0]
print("New size: %d" % len(df))


# Also the fare amount cannot be above 600, it could be because of the incorrect input values 

# In[ ]:


df[df.fare_amount>600]


# As there are only 8 rows we can drop these values. Since the distance between  is of the garbage value. Better to drop these off

# In[ ]:


print("old size: %d" % len(df))
df = df[df.fare_amount <=600]
print("New size: %d" % len(df))


# Passenger count cannot be more then 12 considering everyone as kid, still more then 12 is physically impossible 

# In[ ]:


df = df[df.passenger_count<12]


# **Modelling**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


chunks = pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv", nrows = 5_00,low_memory=False)
rf = RandomForestRegressor(warm_start=True,n_estimators= 20)

chunks = data_cleaning(chunks)
chunks = chunks.drop('key',axis=1)
chunks = chunks[chunks.fare_amount >=0]
chunks = chunks[chunks.fare_amount <=600]
chunks = chunks[chunks.passenger_count<12]
add_new_date_time_features(chunks)
chunks = chunks.drop('pickup_datetime',axis=1)
chunks = map_box(chunks)
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',chunks)
X = chunks.iloc[:,chunks.columns!='fare_amount']
y = chunks['fare_amount'].values  
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
mse = mean_squared_error(y_test,pred)
rmse = math.sqrt(mse)
print (rmse)
print(chunks.shape)


# In[ ]:


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",rf.score(X_test,y_test))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

chunks1 = 5_000_0
i = 0 
z = []
rf = RandomForestRegressor(warm_start=True,n_estimators= 20)
global testP
global pred
for chunks in pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv", chunksize = chunks1,low_memory=False):
    i = i+1
    chunks = data_cleaning(chunks)
    chunks = chunks.drop('key',axis=1)
    chunks = chunks[chunks.fare_amount >=0]
    chunks = chunks[chunks.fare_amount <=600]
    chunks = chunks[chunks.passenger_count<12]
    add_new_date_time_features(chunks)
    chunks = chunks.drop('pickup_datetime',axis=1)
    chunks = map_box(chunks)
    haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',chunks)
    X_train = chunks.iloc[:,chunks.columns!='fare_amount']
    Y_train = chunks['fare_amount'].values    
    
    if(i < 2):
        rf.fit(X_train, Y_train)
        rf.n_estimators += 1
    print("iteration no: "+str(i))    

    if(i==2):
        testP = Y_train
        pred = rf.predict(X_train)
        break


# In[ ]:


from sklearn.metrics import mean_squared_error
import math
mse = mean_squared_error(testP,pred)
rmse = math.sqrt(mse)
print (rmse)


# In[ ]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",rf.score(X_train,testP))


# **Optimisation will happen now**
# The RMSE for our model is approx around 4.5 to 6 
# Now we need to tune our hyperparameters to get a better model 

# In[ ]:


chunks = pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv", nrows = 500,low_memory=False)

chunks.head()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import math
i = 0 

estimators = [ 20 , 60 , 65 ,70 , 75, 80  ]
chunks = pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv", nrows = 5_0000,low_memory=False)
chunks = chunks.drop('key',axis=1)
chunks = data_cleaning(chunks)
chunks = chunks[chunks.fare_amount >=0]
chunks = chunks[chunks.fare_amount <=600]
chunks = chunks[chunks.passenger_count<12]
add_new_date_time_features(chunks)
chunks = chunks.drop('pickup_datetime',axis=1)
chunks = map_box(chunks)
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',chunks)

test_results = []
for estimator in estimators:
    rf = RandomForestRegressor(n_estimators=estimator, n_jobs=-1,warm_start = True)
    
    i = i+1
    X = chunks.iloc[:,chunks.columns!='fare_amount']
    y = chunks['fare_amount'].values  
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    rmse = math.sqrt(mse)
    test_results.append(rmse)
    print("iteration no: "+str(i))    
    
ax = plt.axes()   
ax.plot(estimators,test_results)
plt.show()


# Best n estimator is around 70 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import math
i = 0 

max_features1 = [0.1 , 0.2 , 0.3 , 0.4 , 0.5]
chunks = pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv", nrows = 5_000_0,low_memory=False)
chunks = chunks.drop('key',axis=1)
chunks = data_cleaning(chunks)
chunks = chunks[chunks.fare_amount >=0]
chunks = chunks[chunks.fare_amount <=600]
chunks = chunks[chunks.passenger_count<12]
add_new_date_time_features(chunks)
chunks = chunks.drop('pickup_datetime',axis=1)
chunks = map_box(chunks)
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',chunks)

test_results = []
for features in max_features1:
    rf = RandomForestRegressor(n_estimators= 70, n_jobs=-1,max_features = features)
    
    i = i+1
    X = chunks.iloc[:,chunks.columns!='fare_amount']
    y = chunks['fare_amount'].values  
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    rmse = math.sqrt(mse)
    test_results.append(rmse)
    print("iteration no: "+str(i))    
    


# In[ ]:


ax = plt.axes()   
ax.plot(max_features1,test_results)
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import math
i = 0 

min_samples_leaf1 = [1 , 2 ,3 ,10 , 40]
chunks = pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv", nrows = 5_0,low_memory=False)
chunks = chunks.drop('key',axis=1)
chunks = data_cleaning(chunks)
chunks = chunks[chunks.fare_amount >=0]
chunks = chunks[chunks.fare_amount <=600]
chunks = chunks[chunks.passenger_count<12]
add_new_date_time_features(chunks)
chunks = chunks.drop('pickup_datetime',axis=1)
chunks = map_box(chunks)
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',chunks)

test_results = []
for leaves in min_samples_leaf1:
    rf = RandomForestRegressor(n_estimators= 70, n_jobs=-1,max_features = 0.45 , min_samples_leaf = leaves)
    
    i = i+1
    X = chunks.iloc[:,chunks.columns!='fare_amount']
    y = chunks['fare_amount'].values  
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    rmse = math.sqrt(mse)
    test_results.append(rmse)
    print("iteration no: "+str(i))    
    
ax = plt.axes()   
ax.plot(min_samples_leaf1,test_results)
plt.show()


# tuning hyperparameter a little more

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

chunks1 = 5_00_0
i = 0 
z = []
rf = RandomForestRegressor(n_estimators=80, max_features=0.5, n_jobs=-1,oob_score=True, warm_start =True)
global testP
global pred
for chunks in pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/train.csv", chunksize = chunks1,low_memory=False):
    i = i+1
    chunks = data_cleaning(chunks)
    chunks = chunks.drop('key',axis=1)
    chunks = chunks[chunks.fare_amount >=0]
    chunks = chunks[chunks.fare_amount <=600]
    chunks = chunks[chunks.passenger_count<12]
    add_new_date_time_features(chunks)
    chunks = chunks.drop('pickup_datetime',axis=1)
    chunks = map_box(chunks)
    haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',chunks)
    X_train = chunks.iloc[:,chunks.columns!='fare_amount']
    Y_train = chunks['fare_amount'].values    
    
    if(i < 8001):
        rf.fit(X_train, Y_train)
        rf.n_estimators += 1
    print("iteration no: "+str(i))    

    if(i==8001):
        testP = Y_train
        pred = rf.predict(X_train)
        break


# In[ ]:


from sklearn.externals import joblib 
  
# Save the model as a pickle in a file 
joblib.dump(rf, 'Submission_8000.pkl') 


# In[ ]:


import pickle 
  
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(rf)


# In[ ]:


from sklearn.metrics import mean_squared_error
import math
mse = mean_squared_error(testP,pred)
rmse = math.sqrt(mse)
print (rmse)


# In[ ]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",rf.score(X_train,testP))


# In[ ]:


test = pd.read_csv("/kaggle/input/iiitb2019nyctaxifare/TrainTest/test.csv")


# In[ ]:


test['dropoff_latitude'].fillna(test['dropoff_latitude'].mean(), inplace = True)
test['dropoff_longitude'].fillna(test['dropoff_longitude'].mean(), inplace = True)
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',test)
#test.head()


# In[ ]:


temp = test['key']
temp.to_csv("key.csv", index = False , header = ['key'] )
test = test.drop('key',axis=1)


# In[ ]:


add_new_date_time_features(test)
test = test.drop('pickup_datetime',axis=1)


# In[ ]:


pred = rf.predict(test)
key_temp = pd.read_csv("key.csv")
test['key'] = key_temp
test['predicted_fare_amount'] = pred


# In[ ]:



test.head()


# In[ ]:


result.head()


# In[ ]:


result = test.loc[:, [ 'key','predicted_fare_amount']]
result.to_csv("submit.csv", index = False )


# In[ ]:


read = pd.read_csv("submit.csv")


# In[ ]:


read.head(30)


# In[ ]:


read.shape()

