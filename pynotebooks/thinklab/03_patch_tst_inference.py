#!/usr/bin/env python
# coding: utf-8

# # Inferencing a Time Series Model using Channel Independence PatchTST
# 
# <ul>
# <li>Contributors: IBM AI Research team and IBM Research Technology Education team
# <li>Contact for questions and technical support: IBM.Research.JupyterLab@ibm.com
# <li>Provenance: IBM Research
# <li>Version: 1.0.0
# <li>Release date: 
# <li>Compute requirements: 4 CPU
# <li>Memory requirements: 16 GB
# <li>Notebook set: Time Series Foundation Model
# </ul>

# # Summary
# 
# **Patch Time Series Transformer (PatchTST)** is a new method for long-term forecasting based on Transformer modeling. In PatchTST, a time series is segmented into subseries-level patches that are served as input tokens to Transformer. PatchTST was first proposed in 2023 in [this paper](https://arxiv.org/pdf/2211.14730.pdf). It can achieve state-of-the-art results when compared to other Transformer-based models.
# 
# **Channel Independence PatchTST** is a variant of PatchTST where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series.
# 
# This notebook is the last of three notebooks that should be run in sequence. After running the first notebook, `01_patch_tst_pretrain.ipynb`, a pretrained model was saved in your private storage. The second notebook, `02_patch_tst_fine_tune.ipynb`, loaded the pretrained model and created a fine tuned model, which was also saved in your private storage. This notebook demonstrates inferencing on test data using the fine tuned model. The goal of this demonstration is to forecast the future sensor values (load, oil temperature) of an electric transformer using test data from the ETTh1 benchmark dataset.

# # Table of Contents
# 
# * <a href="#TST3_intro">Channel Independence PatchTST</a>
# * <a href="#TST3_codes">Code Samples</a>
#     * <a href="#TST3_import">Step 1. Imports</a>
#     * <a href="#TST3_pipeln">Step 2. Load model and construct forecasting pipeline</a>
#     * <a href="#TST3_datast">Step 3. Load and prepare datasets </a>
#     * <a href="#TST3_forecs">Step 4. Generate forecasts </a>
#     * <a href="#TST3_perfor">Step 5. Evaluate performance </a>
#     * <a href="#TST3_visual">Step 6. Plot results </a>
# * <a href="#TST3_concl">Conclusion</a>
# * <a href="#TST3_learn">Learn More</a>

# In[ ]:





# <a id="TST3_intro"></a>
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

# <a id="TST3_codes"></a>
# # Code Samples
# 
# This section includes documentation and code samples to demonstrate the use of the toolkit for inferencing.

# <a id="TST3_import"></a>
# ## Step 1. Imports

# In[1]:


import pandas as pd
from transformers.models.patchtst import PatchTSTForPrediction

from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index
from tsfm_public.toolkit.visualization import plot_ts_forecasting

# Customized IBM stylesheet for plots - dark background
# %run '/opt/ibm/visualization/plotly/plotly_template_dark.ipynb'


# <a id="TST3_pipeln"></a>
# ## Step 2. Load model and construct forecasting pipeline
# 
#  Please adjust the following parameters to suit your application:
#  - timestamp_column: column name containing timestamp information, use None if there is no such column
#  - id_columns: List of column names specifying the IDs of different time series. If no ID column exists, use []
#  - forecast_columns: List of columns to be modeled
#  - finetuned_model_path: Path to the finetuned model
#    

# In[2]:


timestamp_column = "date"
id_columns = []
forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
finetuned_model_path = "model/forecasting"


# In[3]:


model = PatchTSTForPrediction.from_pretrained(finetuned_model_path)
# model.model.mask_input = False
forecast_pipeline = TimeSeriesForecastingPipeline(
    model=model,
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    target_columns=forecast_columns,
)
context_length = model.config.context_length

tsp = TimeSeriesPreprocessor.from_pretrained("preprocessor")


# <a id="TST3_datast"></a>
# ## Step 3. Load and prepare datasets
# 
# The specific data loaded here is Electricity Transformer Temperature (ETT) data - including load, oil temperature in an electric transformer. In the next cell, please adjust the following parameters to suit your application:
# 
#  - dataset_path: path to local .csv file, or web address to a csv file for the data of interest. Data is loaded with pandas, so anything supported by pd.read_csv is supported: (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).
#  - test_start_index, test_end_index: the start and end indices in the loaded data which delineate the test data.
# 

# In[4]:


dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
test_start_index = 12 * 30 * 24 + 4 * 30 * 24 - context_length
test_end_index = 12 * 30 * 24 + 8 * 30 * 24


# In[5]:


data = pd.read_csv(
    dataset_path,
    parse_dates=[timestamp_column],
)

test_data = select_by_index(
    data,
    id_columns=id_columns,
    start_index=test_start_index,
    end_index=test_end_index,
)
test_data = tsp.preprocess(test_data)


# In[7]:


test_data.head()


# <a id="TST3_forecs"></a>
# ## Step 4. Generate forecasts
# 
#  Note that the ouput will consist of a Pandas dataframe with the following structure.
#  If you have specified timestamp and/or ID columns they will be included. The forecast
#  columns will be named `{forecast column}_prediction`, for each `{forecast column}` that was
#  specified.
#  Each forecast column will be a vector of values with length equal to the prediction horizon
#  that was specified when the model was trained.

# In[8]:


forecasts = forecast_pipeline(test_data)
forecasts.head()


# <a id="TST3_perfor"></a>
# ## Step 5. Evaluate performance
# 

# In[11]:


try:
    from tsevaluate.multivalue_timeseries_evaluator import CrossTimeSeriesEvaluator

    do_eval = True
except ModuleNotFoundError:
    # tsevaluate (utilities for evaluating multivariate and multi-time series forecasting) will be made available at a later date
    print("tsevaluate module not available.")
    do_eval = False

labels_ = forecasts[id_columns + [timestamp_column] + forecast_columns]
forecasts_ = forecasts.drop(columns=forecast_columns)

if do_eval:
    eval = CrossTimeSeriesEvaluator(
        timestamp_column=timestamp_column,
        prediction_columns=[f"{c}_prediction" for c in forecast_columns],
        label_columns=forecast_columns,
        metrics_spec=["mse", "smape", "rmse", "mae"],
        multioutput="uniform_average",
    )
    eval.evaluate(labels_, forecasts_)


# <a id="TST3_visual"></a>
# ## Step 6. Plot results
# 
# 

# In[12]:


plot_ts_forecasting(
    test_data,
    forecasts_,
    forecast_columns=["HUFL"],
    timestamp_column=timestamp_column,
    periodicity="1h",
    prediction_length=model.config.prediction_length,
    context_length=context_length,
    plot_start=0,
    plot_end=context_length + model.config.prediction_length * 3,
    num_predictions=3,
    plot_stride=model.config.prediction_length,
    title="Forecast",
    fig_size=(1100, 600),
    plot_type="plotly",
    return_image=False,
)


# In[13]:


plot_ts_forecasting(
    test_data,
    forecasts_,
    forecast_columns=["MUFL"],
    timestamp_column=timestamp_column,
    periodicity="1h",
    prediction_length=model.config.prediction_length,
    context_length=context_length,
    plot_start=0,
    plot_end=context_length + model.config.prediction_length * 3,
    num_predictions=3,
    plot_stride=model.config.prediction_length,
    title="Forecast",
    fig_size=(1100, 600),
    plot_type="plotly",
    return_image=False,
)


# <a id="TST3_concl"></a>
# # Conclusion
# 
# This notebook showed how to perform inferencing in a Channel Independence PatchTST model. This is the last of three notebooks that were run in sequence using training and test data from the ETTh1 benchmark dataset, which represents sensor data from an electric transformer.
# 
# The first notebook, `01_patch_tst_pretrain.ipynb`, created a pretrained model that was saved in your private storage. The second notebook, `02_patch_tst_fine_tune.ipynb`, loaded the pretrained model and created a fine tuned model, which was also saved in your private storage. This notebook performed inferencing on the fine tuned model. The goal of this demonstration was to forecast the future sensor values (load, oil temperature) of an electric transformer using test data.

# <a id="TST3_learn"></a>
# # Learn More
# 
# [This paper](https://arxiv.org/pdf/2211.14730.pdf) provides detailed information on Channel Independence PatchTST, including evaluations of its performance on 8 popular datasets, including Weather, Traffic, Electricity, ILI and 4 Electricity Transformer Temperature datasets (ETTh1, ETTh2, ETTm1, ETTm2). These publicly available datasets have been extensively utilized for benchmarking. We featured one of them (ETTh1) in this notebook.
# 
# If you have any questions or wish to schedule a technical deep dive, contact us by email at IBM.Research.JupyterLab@ibm.com.

# © 2023 IBM Corporation
