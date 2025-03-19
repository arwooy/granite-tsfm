#!/usr/bin/env python
# coding: utf-8

# # Pretraining a Time Series Model using Channel Independence PatchTST
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
# This notebook shows how to pretrain a Channel Independence PatchTST model. In this context, pretraining means that the model is trained in a self-supervised way using masking. Individual patches (small segments of the input time series) are masked and the model is trying to reconstruct the missing patches.
# 
# This is the first of three notebooks that should be run in sequence using training and test data from the ETTh1 benchmark dataset, which represents oil temperature in an electric transformer. After running this notebook, a pretrained model will be saved in your private storage. The second notebook, `02_patch_tst_fine_tune.ipynb`, will load the pretrained model and create a fine tuned model, which will be also saved in your private storage. The third notebook, `03_patch_tst_inference.ipynb`, will perform inferencing using the fine tuned model, with a goal of predicting the future sensor values (loads, oil temperature) of an electric transformer.

# # Table of Contents
# 
# * <a href="#TST1_intro">Channel Independence PatchTST</a>
# * <a href="#TST1_codes">Code Samples</a>
#     * <a href="#TST1_import">Step 1. Imports</a>
#     * <a href="#TST1_datast">Step 2. Load and prepare datasets </a>
#     * <a href="#TST1_config">Step 3. Configure the PatchTST model </a>
#     * <a href="#TST1_trainm">Step 4. Train model </a>
# * <a href="#TST1_concl">Conclusion</a>
# * <a href="#TST1_learn">Learn More</a>

# <a id="TST1_intro"></a>
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

# <a id="TST1_codes"></a>
# # Code Samples
# 
# This section includes documentation and code samples to demonstrate the use of the toolkit for pretraining.

# <a id="TST1_import"></a>
# ## Step 1. Imports 

# In[1]:


import pandas as pd
from transformers import (
    PatchTSTConfig,
    PatchTSTForPretraining,
    Trainer,
    TrainingArguments,
)

from tsfm_public.toolkit.dataset import PretrainDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index


#  <a id="TST1_datast"></a>
# ## Step 2. Load and prepare datasets
# 
# 
#  In the next cell, please adjust the following parameters to suit your application:
#  - dataset_path: path to local .csv file, or web address to a csv file for the data of interest. Data is loaded with pandas, so anything supported by
#    `pd.read_csv` is supported: (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).
#  - timestamp_column: column name containing timestamp information, use None if there is no such column
#  - id_columns: List of column names specifying the IDs of different time series. If no ID column exists, use []
#  - forecast_columns: List of columns to be modeled
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


dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
timestamp_column = "date"
id_columns = []
forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

context_length = 512

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

tsp = TimeSeriesPreprocessor(
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    target_columns=forecast_columns,
    scaling=True,
)
tsp.train(train_data)
train_dataset = PretrainDFDataset(
    tsp.preprocess(train_data),
    id_columns=id_columns,
    target_columns=forecast_columns,
    context_length=context_length,
)
eval_dataset = PretrainDFDataset(
    tsp.preprocess(eval_data),
    timestamp_column=timestamp_column,
    target_columns=forecast_columns,
    context_length=context_length,
)


# <a id="TST1_config"></a>
# ## Step 3. Configure the PatchTST model
# 
#  The settings below control the different components in the PatchTST model.
#   - num_input_channels: the number of input channels (or dimensions) in the time series data. This is
#     automatically set to the number for forecast columns.
#   - context_length: As described above, the amount of historical data used as input to the model.
#   - patch_length: The length of the patches extracted from the context window (of length `context_length`).
#   - patch_stride: The stride used when extracting patches from the context window.
#   - mask_ratio: The fraction of input patches that are completely masked for the purpose of pretraining the model.
#   - d_model: Dimension of the transformer layers.
#   - encoder_attention_heads: The number of attention heads for each attention layer in the Transformer encoder.
#   - encoder_layers: The number of encoder layers.
#   - encoder_ffn_dim: Dimension of the intermediate (often referred to as feed-forward) layer in the encoder.
#   - dropout: Dropout probability for all fully connected layers in the encoder.
#   - head_dropout: Dropout probability used in the head of the model.
#   - pooling_type: Pooling of the embedding. `"mean"`, `"max"` and `None` are supported.
#   - channel_attention: Activate channel attention block in the Transformer to allow channels to attend each other.
#   - scaling: Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
#     scaler is set to `"mean"`.
#   - loss: The loss function for the model corresponding to the `distribution_output` head. For parametric
#     distributions it is the negative log likelihood (`"nll"`) and for point estimates it is the mean squared
#     error `"mse"`.
#   - pre_norm: Normalization is applied before self-attention if pre_norm is set to `True`. Otherwise, normalization is
#     applied after residual block.
#   - norm: Normalization at each Transformer layer. Can be `"BatchNorm"` or `"LayerNorm"`.
# 
#  We recommend that you only adjust the values in the next cell.

# In[4]:


patch_length = 12
patch_stride = patch_length


# In[5]:


config = PatchTSTConfig(
    num_input_channels=tsp.num_input_channels,
    context_length=context_length,
    patch_length=patch_length,
    patch_stride=patch_stride,
    mask_ratio=0.4,
    d_model=128,
    encoder_attention_heads=16,
    encoder_layers=3,
    encoder_ffn_dim=512,
    dropout=0.2,
    head_dropout=0.2,
    pooling_type=None,
    channel_attention=False,
    scaling="std",
    loss="mse",
    pre_norm=True,
    norm="batchnorm",
)
pretraining_model = PatchTSTForPretraining(config)


# <a id="TST1_trainm"></a>
# ## Step 4. Train model
# 
#  Trains the PatchTST model based on the Mask-based pretraining strategy. We recommend that the user keep the settings
#  as they are below, with the exception of:
#   - num_train_epochs: The number of training epochs. This may need to be adjusted to ensure sufficient training.
# 

# In[6]:


training_args = TrainingArguments(
    output_dir="./checkpoint/pretrain",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    num_train_epochs=3,  # 50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    logging_strategy="epoch",
    load_best_model_at_end=True,
    # max_steps=10,  # For a quick test
    label_names=["past_values"],
)
pretrainer = Trainer(
    model=pretraining_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

pretrainer.train()


# ## Save the model

# In[7]:


pretrainer.save_model("model/pretrained")


# <a id="TST1_concl"></a>
# # Conclusion
# 
# This notebook showed how to pretrain a Channel Independence PatchTST model. In this context, pretraining means that the model is trained in a self-supervised way using masking. Individual patches (small segments of the input time series) are masked and the model is trying to reconstruct the missing patches.
# 
# This is the first of three notebooks that should be run in sequence using training and test data from the ETTh1 benchmark dataset, which represents sensor data from an electric transformer.
# 
# The above output shows the performance (training loss and validation loss) of the model during the pretraining process. In this case we are using mean squared error (MSE) as a loss function. As the epochs progress we want performance to improve. We would like to see the training and validation losses decrease rapidly for a few epochs and then converge. Validation loss should be relatively close to training loss. Large differences between training and validation losses may be indicative of overfitting (training much lower than validation) or distribution shift between training and validation datasets.

# <a id="TST1_learn"></a>
# # Learn More
# 
# [This paper](https://arxiv.org/pdf/2211.14730.pdf) provides detailed information on Channel Independence PatchTST, including evaluations of its performance on 8 popular datasets, including Weather, Traffic, Electricity, ILI and 4 Electricity Transformer Temperature datasets (ETTh1, ETTh2, ETTm1, ETTm2). These publicly available datasets have been extensively utilized for benchmarking. We featured one of them (ETTh1) in this notebook.
# 
# If you have any questions or wish to schedule a technical deep dive, contact us by email at IBM.Research.JupyterLab@ibm.com.

# © 2023 IBM Corporation
