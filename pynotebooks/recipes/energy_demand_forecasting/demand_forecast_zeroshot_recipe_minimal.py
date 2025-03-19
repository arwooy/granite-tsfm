#!/usr/bin/env python
# coding: utf-8

# # Granite-TimeSeries-TTM   
# 
# TinyTimeMixers (TTMs) are compact pre-trained models for Multivariate Time-Series Forecasting, open-sourced by IBM Research. With less than 1 Million parameters, TTM introduces the notion of the first-ever "tiny" pre-trained models for Time-Series Forecasting. TTM outperforms several popular benchmarks demanding billions of parameters in zero-shot and few-shot forecasting and can easily be fine-tuned for multi-variate forecasts.

# In[1]:


import pathlib

import pandas as pd

from tsfm_public import TimeSeriesForecastingPipeline, TinyTimeMixerForPrediction
from tsfm_public.toolkit.visualization import plot_predictions


# In[2]:


import tsfm_public


tsfm_public.__version__


# ## Initial setup
# 1. Download energy_data.csv.zip and weather_data.csv.zip from https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather
# 2. Place the downloaded files into a folder and update the data_path below

# In[5]:


data_path = pathlib.Path("~/Downloads")


# ## Load and prepare data

# In[6]:


# Download energy_data.csv.zip from https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

dataset_path = data_path / "energy_dataset.csv.zip"
timestamp_column = "time"

target_column = "total load actual"

context_length = 512  # set by the pretrained model we will use

data = pd.read_csv(
    dataset_path,
    parse_dates=[timestamp_column],
)

data = data.ffill()

data = data.iloc[-context_length:,]

print(data.shape)
data.head()


# ## Load pretrained Granite-TimeSeries-TTM model (zero-shot)
# The **TTM** model supports huggingface model interface, allowing easy API for loading the saved models.

# In[7]:


zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-v1", num_input_channels=1
)
model_parameters = sum(p.numel() for p in zeroshot_model.parameters() if p.requires_grad)
print("TTM Model parameters:", model_parameters)


# ### Create a time series forecasting pipeline

# In[8]:


pipeline = TimeSeriesForecastingPipeline(
    zeroshot_model,
    timestamp_column=timestamp_column,
    target_columns=[target_column],
    explode_forecasts=True,
    freq="h",
    id_columns=[],
)
zeroshot_forecast = pipeline(data)
zeroshot_forecast.head()


# ### Plot the results

# In[9]:


plot_predictions(
    input_df=data,
    exploded_predictions_df=zeroshot_forecast,
    freq="h",
    timestamp_column=timestamp_column,
    channel=target_column,
    indices=[-1],
    num_plots=1,
)


# ## Useful links
# 
# TinyTimeMixer paper: https://arxiv.org/abs/2401.03955  
# 
# Granite-TimeSeries-TTM model: https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1  
# 
# Publicly available tools for working with our models: https://github.com/ibm-granite/granite-tsfm

# © 2024 IBM Corporation
