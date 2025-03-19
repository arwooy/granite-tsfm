#!/usr/bin/env python
# coding: utf-8

#  # Patch Time Series Mixer for Transfer Learning across datasets

#  The `PatchTSMixer` model was proposed in [TSMixer: Lightweight MLP-Mixer Model for Multivariate
#  Time Series Forecasting](https://arxiv.org/pdf/2306.09364.pdf) by Vijay Ekambaram, Arindam Jati,
#  Nam Nguyen, Phanwadee Sinthong and Jayant Kalagnanam.
# 
#  `PatchTSMixer` is a time-series foundation modeling approach based on segmentation of time series into subseries-level patches and channel-independence.
# 
#  In this notebook, we will demonstrate the tranfer learning capability of the `PatchTSMixer` model.
#  We will pretrain the model for a forecasting task on a `source` dataset. Then, we will use the
#  pretrained model for a zero-shot forecasting on a `target` dataset. The zero-shot forecasting
#  performance will denote the `test` performance of the model in the `target` domain, without any
#  training on the target domain. Subsequently, we will do linear probing and (then) finetuning of
#  the pretrained model on the `train` part of the target data, and will validate the forecasting
#  performance on the `test` part of the target data.

# In[1]:


# Standard
import os
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


#  ## Set seed

# In[2]:


SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ## Load and prepare datasets
# 
# In the next cell, please adjust the following parameters to suit your application:
# - `PRETRAIN_AGAIN`: Set this to `True` if you want to perform pretraining again. Note that this might take some time depending on the GPU availability. Otherwise, the already pretrained model will be used.
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


PRETRAIN_AGAIN = True
# Download ECL data from https://github.com/zhouhaoyi/Informer2020
dataset_path = "~/Downloads/ECL.csv"
timestamp_column = "date"
id_columns = []

context_length = 512
forecast_horizon = 96
patch_length = 8
num_workers = 16  # Reduce this if you have low number of CPU cores
batch_size = 64  # Adjust according to GPU memory


# In[4]:


if PRETRAIN_AGAIN:
    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )
    forecast_columns = list(data.columns[1:])

    # get split
    num_train = int(len(data) * 0.7)
    num_test = int(len(data) * 0.2)
    num_valid = len(data) - num_train - num_test
    border1s = [
        0,
        num_train - context_length,
        len(data) - num_test - context_length,
    ]
    border2s = [num_train, num_train + num_valid, len(data)]

    train_start_index = border1s[0]  # None indicates beginning of dataset
    train_end_index = border2s[0]

    # we shift the start of the evaluation period back by context length so that
    # the first evaluation timestamp is immediately following the training data
    valid_start_index = border1s[1]
    valid_end_index = border2s[1]

    test_start_index = border1s[2]
    test_end_index = border2s[2]

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


# In[5]:


if PRETRAIN_AGAIN:
    train_dataset = ForecastDFDataset(
        tsp.preprocess(train_data),
        id_columns=id_columns,
        timestamp_column="date",
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    valid_dataset = ForecastDFDataset(
        tsp.preprocess(valid_data),
        id_columns=id_columns,
        timestamp_column="date",
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    test_dataset = ForecastDFDataset(
        tsp.preprocess(test_data),
        id_columns=id_columns,
        timestamp_column="date",
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )


#  ## Configure the PatchTSMixer model
# 
#  The settings below control the different components in the PatchTSMixer model.
#   - `num_input_channels`: the number of input channels (or dimensions) in the time series data. This is
#     automatically set to the number for forecast columns.
#   - `context_length`: As described above, the amount of historical data used as input to the model.
#   - `prediction_length`: This is same as the forecast horizon as decribed above.
#   - `patch_length`: The length of the patches extracted from the context window (of length `context_length``).
#   - `patch_stride`: The stride used when extracting patches from the context window.
#   - `d_model`: Hidden feature dimension of the model.
#   - `num_layers`: The number of model layers.
#   - `dropout`: Dropout probability for all fully connected layers in the encoder.
#   - `head_dropout`: Dropout probability used in the head of the model.
#   - `mode`: PatchTSMixer operating mode. "common_channel"/"mix_channel". Common-channel works in channel-independent mode. For pretraining, use "common_channel".
#   - `scaling`: Per-widow standard scaling. Recommended value: "std".
# 
#  We recommend that you only adjust the values in the next cell.

# In[6]:


if PRETRAIN_AGAIN:
    config = PatchTSMixerConfig(
        context_length=context_length,
        prediction_length=forecast_horizon,
        patch_length=patch_length,
        num_input_channels=len(forecast_columns),
        patch_stride=patch_length,
        d_model=16,
        num_layers=8,
        expansion_factor=2,
        dropout=0.2,
        head_dropout=0.2,
        mode="common_channel",
        scaling="std",
    )
    model = PatchTSMixerForPrediction(config)


#  ## Train model
# 
#  Trains the PatchTSMixer model based on the direct forecasting strategy.

# In[7]:


if PRETRAIN_AGAIN:
    training_args = TrainingArguments(
        output_dir="./checkpoint/patchtsmixer/electricity/pretrain/output/",
        overwrite_output_dir=True,
        learning_rate=0.001,
        num_train_epochs=100,  # For a quick test of this notebook, set it to 1
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        logging_dir="./checkpoint/patchtsmixer/electricity/pretrain/logs/",  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        label_names=["future_values"],
        # max_steps=20,
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0001,  # Minimum improvement required to consider as improvement
    )

    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback],
    )

    # pretrain
    trainer.train()


#  ## Evaluate model on the test set of the `source` domain
#  Note that this is not the target metric to judge in this task.

# In[8]:


if PRETRAIN_AGAIN:
    results = trainer.evaluate(test_dataset)
    print("Test result:")
    print(results)


#  ## Save model

# In[9]:


if PRETRAIN_AGAIN:
    save_dir = "patchtsmixer/electricity/model/pretrain/"
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)


# ## Transfer Learing on `ETTh2` data. All evaluations are on the `test` part of the `ETTh2` data.
# Step 1: Directly evaluate the electricity-pretrained model. This is the zero-shot performance.  
# Step 2: Evalute after doing linear probing.  
# Step 3: Evaluate after doing full finetuning.  

# ### Load ETTh2 data

# In[10]:


dataset = "ETTh2"


# In[11]:


print(f"Loading target dataset: {dataset}")
dataset_path = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset}.csv"
timestamp_column = "date"
id_columns = []
forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
train_start_index = None  # None indicates beginning of dataset
train_end_index = 12 * 30 * 24

# we shift the start of the evaluation period back by context length so that
# the first evaluation timestamp is immediately following the training data
valid_start_index = 12 * 30 * 24 - context_length
valid_end_index = 12 * 30 * 24 + 4 * 30 * 24

test_start_index = 12 * 30 * 24 + 4 * 30 * 24 - context_length
test_end_index = 12 * 30 * 24 + 8 * 30 * 24


# In[12]:


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


# In[13]:


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


# ## Zero-shot forecasting on `ETTh2`

# In[14]:


print("Loading pretrained model")
finetune_forecast_model = PatchTSMixerForPrediction.from_pretrained("patchtsmixer/electricity/model/pretrain/")
print("Done")


# In[15]:


finetune_forecast_args = TrainingArguments(
    output_dir="./checkpoint/patchtsmixer/transfer/finetune/output/",
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
    logging_dir="./checkpoint/patchtsmixer/transfer/finetune/logs/",  # Make sure to specify a logging directory
    load_best_model_at_end=True,  # Load the best model when training ends
    metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
    greater_is_better=False,  # For loss
)

# Create a new early stopping callback with faster convergence properties
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5,  # Number of epochs with no improvement after which to stop
    early_stopping_threshold=0.001,  # Minimum improvement required to consider as improvement
)

finetune_forecast_trainer = Trainer(
    model=finetune_forecast_model,
    args=finetune_forecast_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback],
)

print("\n\nDoing zero-shot forecasting on target data")
result = finetune_forecast_trainer.evaluate(test_dataset)
print("Target data zero-shot forecasting result:")
print(result)


# ## Target data `ETTh2` linear probing
# We can do a quick linear probing on the `train` part of the target data to see any possible `test` performance improvement. 

# In[16]:


# Freeze the backbone of the model
for param in finetune_forecast_trainer.model.model.parameters():
    param.requires_grad = False

print("\n\nLinear probing on the target data")
finetune_forecast_trainer.train()
print("Evaluating")
result = finetune_forecast_trainer.evaluate(test_dataset)
print("Target data head/linear probing result:")
print(result)


# In[17]:


save_dir = f"patchtsmixer/electricity/model/transfer/{dataset}/model/linear_probe/"
os.makedirs(save_dir, exist_ok=True)
finetune_forecast_trainer.save_model(save_dir)

save_dir = f"patchtsmixer/electricity/model/transfer/{dataset}/preprocessor/"
os.makedirs(save_dir, exist_ok=True)
tsp.save_pretrained(save_dir)


# ## Target data `ETTh2` full finetune
# 
# We can do a full model finetune (instead of probing the last linear layer as shown above) on the `train` part of the target data to see a possible `test` performance improvement.

# In[18]:


# Reload the model
finetune_forecast_model = PatchTSMixerForPrediction.from_pretrained("patchtsmixer/electricity/model/pretrain/")
finetune_forecast_trainer = Trainer(
    model=finetune_forecast_model,
    args=finetune_forecast_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback],
)
print("\n\nFinetuning on the target data")
finetune_forecast_trainer.train()
print("Evaluating")
result = finetune_forecast_trainer.evaluate(test_dataset)
print("Target data full finetune result:")
print(result)


# In[19]:


save_dir = f"patchtsmixer/electricity/model/transfer/{dataset}/model/fine_tuning/"
os.makedirs(save_dir, exist_ok=True)
finetune_forecast_trainer.save_model(save_dir)


# In[ ]:




