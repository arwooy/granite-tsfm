#!/usr/bin/env python
# coding: utf-8

# # Quick Start: Running TTM models on gift-eval benchmark
# 
# **Tiny Time Mixers (TTMs)** (accepted in NeurIPS 2024) are **compact and lightweight pre-trained models** for time series forecasting, with sizes ranging from **1 to 5 million parameters**. They are designed for **fast fine-tuning** on target domain datasets.  
# 
# In this script, we demonstrate how to run the **TTM model** on the **GIFT-Eval benchmark** using a **20% few-shot fine-tuning setting**. For more details, see [here](https://github.com/ibm-granite/granite-tsfm/tree/gift/notebooks/hfdemo/tinytimemixer/full_benchmarking/gift_leaderboard). 
# 
# TTM-r2 models have been used in this evaluation. Model card can be found [here](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2).
# 
# Make sure you download the gift-eval benchmark and set the `GIFT-EVAL`
# environment variable correctly before running this script.
# We will use the `Dataset` class to load the data and run the model.
# If you have not already please check out the [dataset.ipynb](./dataset.ipynb)
# notebook to learn more about the `Dataset` class. We are going to just run
# the model on two datasets for brevity. But feel free to run on any dataset
# by changing the `short_datasets` and `med_long_datasets` variables below.

# ## TSFM and TTM Installation
# 
# 1. Clone the [GIFT-Eval repository](https://github.com/SalesforceAIResearch/gift-eval).
# 1. Follow the instruction to set up the GIFT-Eval environment as described [here](https://github.com/SalesforceAIResearch/gift-eval?tab=readme-ov-file#installation).
# 1. This notebook should be placed in the `notebooks` folder of the cloned repository.
# 1. Follow the instructions below to install TSFM. 

# ### Installing `tsfm`
# 
# The TTM source codes will be installed from the [granite-tsfm repository](https://github.com/ibm-granite/granite-tsfm).
# Note that `granite-tsfm` installs `pandas==2.2.3` but GIFT-EVAL requires `pandas==2.0.0`.
# Hence, after installing TTM from `granite-tsfm`, we forece reinstall `pandas==2.0.0`.
# 
# 
# Run the following code once to install granite-tsfm in your working python environment.
# 

# In[1]:


import os


if not os.path.exists("granite-tsfm"):
    get_ipython().system('git clone git@github.com:ibm-granite/granite-tsfm.git')
    get_ipython().run_line_magic('cd', 'granite-tsfm')
    get_ipython().system('pwd')
    # Switch to the desired branch
    get_ipython().system('git switch gift')
    get_ipython().system(' pip install ".[notebooks]"')
    get_ipython().system(' pip install pandas==2.0.0')
    get_ipython().run_line_magic('cd', '..')
else:
    print("Folder 'granite-tsfm' already exists. Skipping git clone.")


# ## Imports

# In[2]:


# All Required Imports
import csv
import json
import sys

import pandas as pd
from dotenv import load_dotenv
from gift_eval.data import Dataset
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality


# ### Add `TTMGluonTSPredictor` to `PYTHONPATH`

# In[3]:


sys.path.append(os.path.realpath("granite-tsfm/notebooks/hfdemo/tinytimemixer/full_benchmarking/"))
from gift_leaderboard.src.ttm_gluonts_predictor import (
    TTM_MAX_FORECAST_HORIZON,
    TTMGluonTSPredictor,
)
from gift_leaderboard.src.utils import get_args, set_seed


# ## Set output directory and seed

# In[4]:


args = get_args()

# Set out dir path
OUT_DIR = f"../results/{args.out_dir}"

# Add arguments
SEED = 42

# set seed
set_seed(SEED)
# Load environment variables
load_dotenv()
# Ensure the output directory exists
os.makedirs(OUT_DIR, exist_ok=True)


# ## Dataset

# In[5]:


# short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
short_datasets = "us_births/D saugeenday/M"

# med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = ""

# Get union of short and med_long datasets
all_datasets = sorted(set(short_datasets.split() + med_long_datasets.split()))

dataset_properties_map = json.load(open("dataset_properties.json"))


# ## Metrics

# In[6]:


# Instantiate the metrics
metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(forecast_type="mean"),
    MAE(forecast_type=0.5),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]


# ## Evaluation
# 
# 
# Now that we have our predictor class `TTMGluonTSPredictor` imported,
# we can use it to fine-tune and predict on the gift-eval benchmark datasets.
# We will use the `train` function to finetune the TTM model, and
# `evaluate_model` function to evaluate the model.
# The `evaluate_model` function is a helper function to evaluate the 
# model on the test data and return the results in a dictionary.
# 
# We are going to follow the naming conventions explained in the
# [README](../README.md) file to store the results in a csv file
# called `all_results.csv` under the `results/ttm` folder.
# 
# The first column in the csv file is the dataset config name which
# is a combination of the dataset name, frequency and the term:
# 
# ```python
# f"{dataset_name}/{freq}/{term}"
# ```

# ### Define output file paths

# In[7]:


# Define the path for the CSV file
csv_file_path = os.path.join(OUT_DIR, "all_results.csv")

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

if not os.path.exists(csv_file_path):
    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(
            [
                "dataset",
                "model",
                "eval_metrics/MSE[mean]",
                "eval_metrics/MSE[0.5]",
                "eval_metrics/MAE[mean]",
                "eval_metrics/MAE[0.5]",
                "eval_metrics/MASE[0.5]",
                "eval_metrics/MAPE[0.5]",
                "eval_metrics/sMAPE[0.5]",
                "eval_metrics/MSIS",
                "eval_metrics/RMSE[mean]",
                "eval_metrics/NRMSE[mean]",
                "eval_metrics/ND[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
                "domain",
                "num_variates",
                "horizon",
                "ttm_context_len",
                "available_context_len",
                "finetune_success",
                "finetune_train_num_samples",
                "finetune_valid_num_samples",
            ]
        )

df_res = pd.read_csv(csv_file_path)
done_datasets = df_res["dataset"].values
print("Done datasets")
print(done_datasets)


# ### Run over all defined datasets

# In[8]:


for ds_name in all_datasets:
    set_seed(SEED)
    terms = ["short", "medium", "long"]
    for term in terms:
        if (term == "medium" or term == "long") and ds_name not in med_long_datasets.split():
            continue

        print(f"Processing dataset: {ds_name}, term: {term}")

        if "/" in ds_name:
            ds_key = ds_name.split("/")[0]
            ds_freq = ds_name.split("/")[1]
            ds_key = ds_key.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = ds_name.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
            ds_freq = dataset_properties_map[ds_key]["frequency"]
        ds_config = f"{ds_key}/{ds_freq}/{term}"

        if ds_config in done_datasets:
            print(f"Done with {ds_config}. Skipping...")
            continue

        dataset = Dataset(name=ds_name, term=term, to_univariate=False)
        season_length = get_seasonality(dataset.freq)

        print(f"Dataset: {ds_name}, Freq = {dataset.freq}, H = {dataset.prediction_length}")

        # Get suitable context length for TTM for this dataset
        all_lengths = []
        for x in dataset.test_data:
            if len(x[0]["target"].shape) == 1:
                all_lengths.append(len(x[0]["target"]))
                num_channels = 1
            else:
                all_lengths.append(x[0]["target"].shape[1])
                num_channels = x[0]["target"].shape[0]

        min_context_length = min(all_lengths)
        print(
            "Minimum context length among all time series in this dataset =",
            min_context_length,
        )

        # Set channel indices
        num_prediction_channels = num_channels
        prediction_channel_indices = list(range(num_channels))

        # Check existence of "past_feat_dynamic_real"
        past_feat_dynamic_real_exist = False
        if args.use_exogs and "past_feat_dynamic_real" in x[0].keys():
            num_exogs = x[0]["past_feat_dynamic_real"].shape[0]
            print(f"Data has `past_feat_dynamic_real` features of size {num_exogs}.")
            num_channels += num_exogs
            past_feat_dynamic_real_exist = True

        if dataset.prediction_length > TTM_MAX_FORECAST_HORIZON:
            # predict all channels, needed for recursive forecast
            prediction_channel_indices = list(range(num_channels))

        print("prediction_channel_indices =", prediction_channel_indices)

        # For very short series, force short context window creatiio for finetuning
        if term == "short":
            force_short_context = args.force_short_context
        else:
            force_short_context = False

        # Instantiate the TTM GluonTS Predictor with the minimum context length in the dataset
        # The predictor will automatically choose the suitable context and forecast length
        # of the TTM model.
        predictor = TTMGluonTSPredictor(
            context_length=min_context_length,
            prediction_length=dataset.prediction_length,
            model_path=args.model_path,
            test_data_label=dataset.test_data.label,
            random_seed=SEED,
            term=term,
            ds_name=ds_name,
            out_dir=OUT_DIR,
            scale=True,
            upper_bound_fewshot_samples=args.upper_bound_fewshot_samples,
            force_short_context=force_short_context,
            min_context_mult=args.min_context_mult,
            past_feat_dynamic_real_exist=past_feat_dynamic_real_exist,
            num_prediction_channels=num_prediction_channels,
            freq=dataset.freq,
            use_valid_from_train=args.use_valid_from_train,
            insample_forecast=args.insample_forecast,
            insample_use_train=args.insample_use_train,
            # TTM kwargs
            head_dropout=args.head_dropout,
            decoder_mode=args.decoder_mode,
            num_input_channels=num_channels,
            huber_delta=args.huber_delta,
            quantile=args.quantile,
            loss=args.loss,
            prediction_channel_indices=prediction_channel_indices,
        )

        print(f"Number of channels in the dataset {ds_name} =", num_channels)
        if args.batch_size is None:
            batch_size = None
            optimize_batch_size = True
        else:
            batch_size = args.batch_size
            optimize_batch_size = False
        print("Batch size is set to", batch_size)

        finetune_train_num_samples = 0
        finetune_valid_num_samples = 0
        try:
            # finetune the model on the train split
            predictor.train(
                train_dataset=dataset.training_dataset,
                valid_dataset=dataset.validation_dataset,
                batch_size=batch_size,
                optimize_batch_size=optimize_batch_size,
                freeze_backbone=args.freeze_backbone,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                fewshot_fraction=args.fewshot_fraction,
                fewshot_location=args.fewshot_location,
                automate_fewshot_fraction=args.automate_fewshot_fraction,
                automate_fewshot_fraction_threshold=args.automate_fewshot_fraction_threshold,
            )
            finetune_success = True
            finetune_train_num_samples = predictor.train_num_samples
            finetune_valid_num_samples = predictor.valid_num_samples
        except Exception as e:
            print("Error in finetune workflow. Error =", e)
            print("Fallback to zero-shot performance.")
            finetune_success = False

        # Evaluate
        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )

        # Append the results to the CSV file
        with open(csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    ds_config,
                    "TTM",
                    res["MSE[mean]"][0],
                    res["MSE[0.5]"][0],
                    res["MAE[mean]"][0],
                    res["MAE[0.5]"][0],
                    res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0],
                    res["sMAPE[0.5]"][0],
                    res["MSIS"][0],
                    res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0],
                    res["ND[0.5]"][0],
                    res["mean_weighted_sum_quantile_loss"][0],
                    dataset_properties_map[ds_key]["domain"],
                    dataset_properties_map[ds_key]["num_variates"],
                    dataset.prediction_length,
                    predictor.ttm.config.context_length,
                    min_context_length,
                    finetune_success,
                    finetune_train_num_samples,
                    finetune_valid_num_samples,
                ]
            )

            print(f"Results for {ds_name} have been written to {csv_file_path}")


# In[9]:


# Results
df = pd.read_csv(f"{OUT_DIR}/all_results.csv")
df = df.sort_values(by="dataset")
display(
    df[
        [
            "dataset",
            "eval_metrics/MASE[0.5]",
            "eval_metrics/NRMSE[mean]",
            "eval_metrics/mean_weighted_sum_quantile_loss",
        ]
    ]
)


# In[ ]:




