#!/usr/bin/env python
# coding: utf-8

# # Getting started with `PatchTSMixer`
# ## Direct forecasting example
# 
# This notebooke demonstrates the usage of a `PatchTSMixer` model for a multivariate time series forecasting task. This notebook has a dependecy on HuggingFace [transformers](https://github.com/huggingface/transformers) repo. For details related to model architecture, refer to the [TSMixer paper](https://arxiv.org/abs/2306.09364).

# In[1]:


# Standard
import random

import numpy as np
import pandas as pd
import torch

# Third Party
from transformers import (
    EarlyStoppingCallback,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    Trainer,
    TrainingArguments,
)

# First Party
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index


# In[2]:


# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ## Load and prepare datasets
# 
# In the next cell, please adjust the following parameters to suit your application:
# - `dataset_path`: path to local .csv file, or web address to a csv file for the data of interest. Data is loaded with pandas, so anything supported by
# `pd.read_csv` is supported: (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).
# - `timestamp_column`: column name containing timestamp information, use None if there is no such column
# - `id_columns`: List of column names specifying the IDs of different time series. If no ID column exists, use []
# - `forecast_columns`: List of columns to be modeled
# - `context_length`: The amount of historical data used as input to the model. Windows of the input time series data with length equal to
# context_length will be extracted from the input dataframe. In the case of a multi-time series dataset, the context windows will be created
# so that they are contained within a single time series (i.e., a single ID).
# - `forecast_horizon`: Number of time stamps to forecast in future.
# - `train_start_index`, `train_end_index`: the start and end indices in the loaded data which delineate the training data.
# - `valid_start_index`, `valid_end_index`: the start and end indices in the loaded data which delineate the validation data.
# - `test_start_index`, `test_end_index`: the start and end indices in the loaded data which delineate the test data.
# - `patch_length`: The patch length for the `PatchTSMixer` model. Recommended to have a value so that `context_length` is divisible by it.
# - `num_workers`: Number of dataloder workers in pytorch dataloader.
# - `batch_size`: Batch size. 
# The data is first loaded into a Pandas dataframe and split into training, validation, and test parts. Then the pandas dataframes are converted
# to the appropriate torch dataset needed for training.

# In[3]:


dataset = "ETTh1"
num_workers = 8  # Reduce this if you have low number of CPU cores
batch_size = 32  # Reduce if not enough GPU memory available
context_length = 512
forecast_horizon = 96
patch_length = 8


# In[4]:


print(f"Loading target dataset: {dataset}")
dataset_path = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset}.csv"
timestamp_column = "date"
id_columns = []
forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
train_start_index = None  # None indicates beginning of dataset
train_end_index = 12 * 30 * 24

# we shift the start of the validation/test period back by context length so that
# the first validation/test timestamp is immediately following the training data
valid_start_index = 12 * 30 * 24 - context_length
valid_end_index = 12 * 30 * 24 + 4 * 30 * 24

test_start_index = 12 * 30 * 24 + 4 * 30 * 24 - context_length
test_end_index = 12 * 30 * 24 + 8 * 30 * 24


# In[5]:


data = pd.read_csv(
    dataset_path,
    parse_dates=[timestamp_column],
)

train_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=train_start_index,
    end_index=train_end_index,
)
valid_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=valid_start_index,
    end_index=valid_end_index,
)
test_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=test_start_index,
    end_index=test_end_index,
)

tsp = TimeSeriesPreprocessor(
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    target_columns=forecast_columns,
    scaling=True,
)
tsp.train(train_data)


# In[6]:


train_dataset = ForecastDFDataset(
    tsp.preprocess(train_data),
    id_columns=id_columns,
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)
valid_dataset = ForecastDFDataset(
    tsp.preprocess(valid_data),
    id_columns=id_columns,
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)
test_dataset = ForecastDFDataset(
    tsp.preprocess(test_data),
    id_columns=id_columns,
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)


# ## Testing with a `PatchTSMixer` model that was trained on the training part of the `ETTh1` data
# 
# A pre-trained model (on `ETTh1` data) is available at [ibm-granite/granite-timeseries-patchtsmixer](https://huggingface.co/ibm-granite/granite-timeseries-patchtsmixer).

# In[8]:


print("Loading pretrained model")
inference_forecast_model = PatchTSMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-patchtsmixer")
print("Done")


# In[9]:


inference_forecast_trainer = Trainer(
    model=inference_forecast_model,
)

print("\n\nDoing testing on Etth1/test data")
result = inference_forecast_trainer.evaluate(test_dataset)
print(result)


# ## If we want to train from scratch
# 
# Adjust the following model parameters according to need.
# - `d_model` (`int`, *optional*, defaults to 8):
#     Hidden dimension of the model. Recommended to set it as a multiple of patch_length (i.e. 2-8X of
#     patch_len). Larger value indicates more complex model.
# - `expansion_factor` (`int`, *optional*, defaults to 2):
#     Expansion factor to use inside MLP. Recommended range is 2-5. Larger value indicates more complex model.
# - `num_layers` (`int`, *optional*, defaults to 3):
#     Number of layers to use. Recommended range is 3-15. Larger value indicates more complex model.

# In[10]:


config = PatchTSMixerConfig(
    context_length=context_length,
    prediction_length=forecast_horizon,
    patch_length=patch_length,
    num_input_channels=len(forecast_columns),
    patch_stride=patch_length,
    d_model=48,
    num_layers=3,
    expansion_factor=3,
    dropout=0.5,
    head_dropout=0.7,
    mode="common_channel",  # change it `mix_channel` if we need to explicitly model channel correlations
    scaling="std",
)
model = PatchTSMixerForPrediction(config=config)


# In[11]:


train_args = TrainingArguments(
    output_dir="./checkpoint/patchtsmixer/direct/train/output/",
    overwrite_output_dir=True,
    learning_rate=0.0001,
    num_train_epochs=100,
    do_eval=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    dataloader_num_workers=num_workers,
    report_to="tensorboard",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=3,
    logging_dir="./checkpoint/patchtsmixer/direct/train/logs/",  # Make sure to specify a logging directory
    load_best_model_at_end=True,  # Load the best model when training ends
    metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
    greater_is_better=False,  # For loss
    label_names=["future_values"],
)

# Create a new early stopping callback with faster convergence properties
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5,  # Number of epochs with no improvement after which to stop
    early_stopping_threshold=0.001,  # Minimum improvement required to consider as improvement
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback],
)

print("\n\nDoing forecasting training on Etth1/train")
trainer.train()


# In[12]:


trainer.evaluate(test_dataset)


# ## If we want to train from scratch for a few specific forecast channels

# In[14]:


forecast_channel_indices = [
    -4,
    -1,
]  # add the channel indices (i.e., the column number) for which the model should forecast


# In[15]:


config = PatchTSMixerConfig(
    context_length=context_length,
    prediction_length=forecast_horizon,
    patch_length=patch_length,
    num_input_channels=len(forecast_columns),
    patch_stride=patch_length,
    d_model=48,
    num_layers=3,
    expansion_factor=3,
    dropout=0.5,
    head_dropout=0.7,
    mode="common_channel",
    scaling="std",
    prediction_channel_indices=forecast_channel_indices,
)
model = PatchTSMixerForPrediction(config=config)


# In[16]:


trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback],
)

print("\n\nDoing forecasting training on Etth1/train")
trainer.train()


# In[17]:


trainer.evaluate(test_dataset)


# #### Sanity check: Compute number of forecasting channels

# In[18]:


output = trainer.predict(test_dataset)


# In[19]:


output.predictions[0].shape

