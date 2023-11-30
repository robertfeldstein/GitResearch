# GitResearch
This repository covers the lightning research performed with Professor Joe Guinness at Cornell University.


In these notebooks we cover how to extract data from NOAA's GOES-R satellites, including its Geostationary Lightning Mapper (GLM) product. We investigate some of the data surrounding lightning strikes, and hope to build a machine learning model which can predict lightning strikes ahead of time. 

## Breakdown of Files

goesaws.py

The goesaws.py file contains many helper functions which are used in the upcoming Jupyter Notebooks. In particular, it contains all of the functions which connect to the NOAA AWS server which actually pulls data from the GOES-R satellites. It also contains methods for converting the radian data into lat-lon, methods for plotting the data using matplotlib, and methods for making GIFs. 

dataset.py

The dataset.py file contains functions for transfoming GOES-R satellite data into a pandas dataframe. It takes a JSON file (with examples shown) in order to produce a dataset for a specific hour of a day. 

FeatureEngineering.ipynb

We tried making additional features to add to the dataset that might be better at predicting lightning. 

LightningPlotter.ipynb

This notebook plots still images of GOES-R data. 

LightningEDA.ipynb

In this notebook we train several machine learning binary classifiers that attempt to use the weather data to predict lightning. 

KerasTensorFlow.ipynb

This notebook trains a TensorFlow deep learning model. At its current stage it does not outperform XGBoost regression. 
