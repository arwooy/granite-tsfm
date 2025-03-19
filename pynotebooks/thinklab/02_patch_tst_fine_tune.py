#!/usr/bin/env python
# coding: utf-8

# # Fine Tuning a Time Series Model using Channel Independence PatchTST
# 
# <ul>
# <li>Contributors: IBM AI Research team and IBM Research Technology Education team
# <li>Contact for questions and technical support: IBM.Research.JupyterLab@ibm.com
# <li>Provenance: IBM Research
# <li>Version: 1.0.0
# <li>Release date: 
# <li>Compute requirements: 4 CPU (preferrably 1 GPU)
# <li>Memory requirements: 16 GB
# <li>Notebook set: Time Series Foundation Model
# </ul>

# # Summary
# 
# **Patch Time Series Transformer (PatchTST)** is a new method for long-term forecasting based on Transformer modeling. In PatchTST, a time series is segmented into subseries-level patches that are served as input tokens to Transformer. PatchTST was first proposed in 2023 in [this paper](https://arxiv.org/pdf/2211.14730.pdf). It can achieve state-of-the-art results when compared to other Transformer-based models.
# 
# **Channel Independence PatchTST** is a variant of PatchTST where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series.
# 
# This notebook shows how to fine tune a Channel Independence PatchTST model in a supervised way. Fine tuning will be set up to only affect the last linear layer of the model -- this is called *linear probing*. The model is trained using patches extracted from a longer segment of the time series as input, with the future values as a target.
# 
# This is the second of three notebooks that should be run in sequence using training and test data from the ETTh1 benchmark dataset, which represents oil temperature in an electric transformer. After running the first notebook, `01_patch_tst_pretrain.ipynb`, a pretrained model was saved in your private storage. This notebook will load the pretrained model and create a fine tuned model, which will be also saved in your private storage. The third notebook, `03_patch_tst_inference.ipynb`, will perform inferencing using the fine tuned model, with a goal of predicting the future temperature of the oil in the electric transformer.
# 
# *Maybe add a picture of the PatchTST with forecasting head?*

# # Table of Contents
# 
# * <a href="#TST2_intro">Channel Independence PatchTST</a>
# * <a href="#TST2_codes">Code Samples</a>
#     * <a href="#TST2_import">Step 1. Imports</a>
#     * <a href="#TST2_datast">Step 2. Load and prepare datasets </a>
#     * <a href="#TST2_config">Step 3. Configure the PatchTST model </a>
#     * <a href="#TST2_modelp">Step 4. Load model and freeze base model parameters </a>
#     * <a href="#TST2_ftunem">Step 5. Fine-tune the model </a>
# * <a href="#TST2_concl">Conclusion</a>
# * <a href="#TST2_learn">Learn More</a>

# <a id="TST2_intro"></a>
# # Channel Independence PatchTST
# 
# **Channel Independence PatchTST** is an efficient design of Transformer-based models for multivariate time series forecasting and self-supervised representation learning. It is demonstrated in the following diagram. It is based on two key components:
# 
# - segmentation of time series into subseries-level patches that are served as input tokens to Transformer
# 
# - channel independence where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series.
# 
# Patching design naturally has three-fold benefit: local semantic information is retained in the embedding; computation and memory usage of the attention maps are quadratically reduced given the same look-back window; and the model can attend longer history.
# 
# Channel independence allows each time series to have its own embedding and attention maps while sharing the same model parameters across different channels.
# 
# <div> <img src="./data/figures/patchTST.png" alt="Drawing" style="width: 600px;"/></div>

# <a id="TST2_codes"></a>
# # Code Samples
# 
# This section includes documentation and code samples to demonstrate the use of the toolkit for fine tuning.

# <a id="TST2_import"></a>
# ## Step 1. Imports

# In[1]:


import pandas as pd
from transformers import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
)

from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index


# <a id="TST2_datast"></a>
# ## Step 2. Load and prepare datasets
# 
#  In the next cell, please adjust the following parameters to suit your application:
#  - dataset_path: path to local .csv file, or web address to a csv file for the data of interest. Data is loaded with pandas, so anything supported by
#    `pd.read_csv` is supported: (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).
#  - timestamp_column: column name containing timestamp information, use None if there is no such column
#  - id_columns: List of column names specifying the IDs of different time series. If no ID column exists, use []
#  - forecast_columns: List of columns to be modeled
#  - prediction_length: Specifies how many timepoints should be forecasted
#  - context_length: The amount of historical data used as input to the model. Windows of the input time series data with length equal to
#    context_length will be extracted from the input dataframe. In the case of a multi-time series dataset, the context windows will be created
#    so that they are contained within a single time series (i.e., a single ID).
#  - train_start_index, train_end_index: the start and end indices in the loaded data which delineate the training data.
#  - eval_start_index, eval_end_index: the start and end indices in the loaded data which delineate the evaluation data.
# 
#  The data is first loaded into a Pandas dataframe and split into training and evaluation parts. Then the pandas dataframes are converted
#  to the appropriate torch dataset needed for training.
#  
#  The specific data loaded here is Electricity Transformer Temperature (ETT) data - including load, oil temperature in an electric transformer.

# In[2]:


dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
timestamp_column = "date"
id_columns = []
forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

prediction_length = 96

pretrained_model_path = "model/pretrained"

# load pretrained model config, to access some previously defined parameters
pretrained_config = PatchTSTConfig.from_pretrained(pretrained_model_path)
context_length = pretrained_config.context_length  # use pretrained_config.context_length to match pretrained model

train_start_index = None  # None indicates beginning of dataset
train_end_index = 12 * 30 * 24

# we shift the start of the evaluation period back by context length so that
# the first evaluation timestamp is immediately following the training data
eval_start_index = 12 * 30 * 24 - context_length
eval_end_index = 12 * 30 * 24 + 4 * 30 * 24


# In[3]:


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
eval_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=eval_start_index,
    end_index=eval_end_index,
)

print(data.head())


# In[4]:


tsp = TimeSeriesPreprocessor(
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    target_columns=forecast_columns,
    scaling=True,
)
tsp.train(train_data)
train_dataset = ForecastDFDataset(
    tsp.preprocess(train_data),
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=prediction_length,
)
eval_dataset = ForecastDFDataset(
    tsp.preprocess(eval_data),
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=prediction_length,
)


# <a id="TST2_config"></a>
# ## Step 3. Configure the PatchTST model
# 
#  The PatchTSTConfig is created in the next cell. This leverages the configuration that
#  is already present in the pretrained model, and adds the parameters necessary for the
#  forecasting task. This includes:
#  - context_length: As described above, the amount of historical data used as input to the model.
#  - num_input_channels: The number of input channels. In this case, it is set equal to the n
#    number of dimensions we intend to forecast.
#  - prediction_length: Prediction horizon for the forecasting task, as set above.

# In[5]:


pred_config = PatchTSTConfig.from_pretrained(
    pretrained_model_path,
    context_length=context_length,
    num_input_channels=tsp.num_input_channels,
    prediction_length=prediction_length,
    do_mask_input=False,
)


# <a id="TST2_modelp"></a>
# ## Step 4. Load model and freeze base model parameters
# 
#  The follwoing cell loads the pretrained model and then freezes parameters in the base
#  model. You will likely see a warning about weights not being initialized from the model
#  checkpoint; this message is expected since the forecasting model has a head with weights
#  which have not yet been trained.

# In[6]:


forecasting_model = PatchTSTForPrediction.from_pretrained(
    "model/pretrained",
    config=pred_config,
    ignore_mismatched_sizes=True,
)
# This freezes the base model parameters
# for param in forecasting_model.base_model.parameters():
#     param.requires_grad = False


# <a id="TST2_ftunem"></a>
# ## Step 5. Fine-tune the model
# 
#  Fine-tunes the PatchTST model using the pretrained base model loaded above. We recommend that the user keep the settings
#  as they are below, with the exception of:
#   - num_train_epochs: The number of training epochs. This may need to be adjusted to ensure sufficient training.
# 

# In[7]:


training_args = TrainingArguments(
    output_dir="./checkpoint/forecast",
    per_device_train_batch_size=8,  # defaults to 8
    per_device_eval_batch_size=64,  # defaults to 8
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    logging_strategy="epoch",
    load_best_model_at_end=True,
    # max_steps=10,  # For a quick test
    label_names=["future_values"],
)

forecasting_trainer = Trainer(
    model=forecasting_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

forecasting_trainer.train()
forecasting_trainer.save_model("model/forecasting")
tsp.save_pretrained("preprocessor")


# <a id="TST2_concl"></a>
# # Conclusion
# 
# This notebook showed how to fine tune a Channel Independence PatchTST model in a supervised way. Fine tuning was set up to only affect the last linear layer of the model (linear probing). The model was trained using patches extracted from a longer segment of the time series as input, with the future values as a target.
# 
# This is the second of three notebooks that should be run in sequence using training and test data from the ETTh1 benchmark dataset, which represents sensor data from an electric transformer.
# 
# The above output shows the performance (training loss and validation loss) of the model during the fine tuning process. In this case we are using mean squared error (MSE) as a loss function. As the epochs progress we want performance to improve. We would like to see the training and validation losses decrease rapidly for a few epochs and then converge. Validation loss should be relatively close to training loss. Large differences between training and validation losses may be indicative of overfitting (training much lower than validation) or distribution shift between training and validation datasets.

# <a id="TST2_learn"></a>
# # Learn More
# 
# [This paper](https://arxiv.org/pdf/2211.14730.pdf) provides detailed information on Channel Independence PatchTST, including evaluations of its performance on 8 popular datasets, including Weather, Traffic, Electricity, ILI and 4 Electricity Transformer Temperature datasets (ETTh1, ETTh2, ETTm1, ETTm2). These publicly available datasets have been extensively utilized for benchmarking. We featured one of them (ETTh1) in this notebook.
# 
# If you have any questions or wish to schedule a technical deep dive, contact us by email at IBM.Research.JupyterLab@ibm.com.

# © 2023 IBM Corporation
