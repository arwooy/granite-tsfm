#!/usr/bin/env python
# coding: utf-8

# # TTM zero-shot and few-shot benchmarking on multiple datasets
# 
# **Using TTM-1024-96 model.**
# 
# Pre-trained TTM models will be fetched from the [Granite-TTM-R2 Model Card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2).
# 
# For details, visit the [Hugging Face TTM Model Repository](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2).
# 
# 1. IBM Granite TTM-R1 pre-trained models can be found here: [Granite-TTM-R1 Model Card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1)
# 2. IBM Granite TTM-R2 pre-trained models can be found here: [Granite-TTM-R2 Model Card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)
# 3. Research-use (non-commercial use only) TTM-R2 pre-trained models can be found here: [Research-Use-TTM-R2](https://huggingface.co/ibm-research/ttm-research-r2)

# ## Imports

# In[1]:


import math
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK

from tsfm_public import TrackingCallback, count_parameters, load_dataset
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions


warnings.filterwarnings("ignore")


# ## Important arguments

# In[2]:


# Set seed
SEED = 42
set_seed(SEED)

# Specify model parameters
context_length = 1024
forecast_length = 96
freeze_backbone = True

# Other args
EPOCHS = 50
NUM_WORKERS = 16

# Make sure all the datasets in the following `list_datasets` are
# saved in the `DATA_ROOT_PATH` folder. Or, change it accordingly.
# Refer to the load_datasets() function
# in notebooks/hfdemo/tinytimemixer/utils/ttm_utils.py
# to see how it is used.
DATA_ROOT_PATH = "/dccstor/tsfm23/datasets/"

# This is where results will be saved
OUT_DIR = f"ttm-r2_results_benchmark_{context_length}_{forecast_length}/"


# ## List of benchmark datasets (TTM was not pre-trained on any of these)

# In[3]:


list_datasets = [
    "etth1",
    "etth2",
    "ettm1",
    "ettm2",
    "weather",
    "electricity",
    "traffic",
]


# ## Set model path

# In[4]:


hf_model_path = "ibm-granite/granite-timeseries-ttm-r2"


# ## Main benchmarking loop

# In[5]:


all_results = {
    "dataset": [],
    "zs_mse": [],
    "fs5_mse": [],
    "zs_eval_time": [],
    "fs5_mean_epoch_time": [],
    "fs5_total_train_time": [],
    "fs5_best_val_metric": [],
}
# Loop over data
for DATASET in list_datasets:
    print()
    print("=" * 100)
    print(
        f"Running zero-shot/few-shot for TTM-{context_length} on dataset = {DATASET}, forecast_len = {forecast_length}"
    )
    print(f"Model will be loaded from {hf_model_path}")
    SUBDIR = f"{OUT_DIR}/{DATASET}"

    # Set batch size
    if DATASET == "traffic":
        BATCH_SIZE = 8
    elif DATASET == "electricity":
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 64

    # Data prep: Get dataset
    _, _, dset_test = load_dataset(DATASET, context_length, forecast_length, dataset_root_path=DATA_ROOT_PATH)

    #############################################################
    ##### Use the pretrained model in zero-shot forecasting #####
    #############################################################
    # Load model
    zeroshot_model = get_model(hf_model_path, context_length=context_length, prediction_length=forecast_length)

    # zeroshot_trainer
    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(
            output_dir=f"{SUBDIR}/zeroshot",
            per_device_eval_batch_size=BATCH_SIZE,
            seed=SEED,
        ),
        eval_dataset=dset_test,
    )

    # evaluate = zero-shot performance
    print("+" * 20, "Test MSE zero-shot", "+" * 20)
    zeroshot_output = zeroshot_trainer.evaluate(dset_test)
    print(zeroshot_output)
    print("+" * 60)
    all_results["zs_eval_time"].append(zeroshot_output["eval_runtime"])

    # Plot
    plot_predictions(
        model=zeroshot_trainer.model,
        dset=dset_test,
        plot_dir=SUBDIR,
        num_plots=10,
        plot_prefix="test_zeroshot",
        channel=0,
    )
    plt.close()

    # write results
    all_results["dataset"].append(DATASET)
    all_results["zs_mse"].append(zeroshot_output["eval_loss"])

    ################################################################
    ## Use the pretrained model in few-shot 5% and 10% forecasting #
    ################################################################
    for fewshot_percent in [5]:
        # Set learning rate
        learning_rate = None  # `None` value indicates that the optimal_lr_finder() will be used

        print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)
        # Data prep: Get dataset
        dset_train, dset_val, dset_test = load_dataset(
            DATASET,
            context_length,
            forecast_length,
            fewshot_fraction=fewshot_percent / 100,
            dataset_root_path=DATA_ROOT_PATH,
        )

        # change head dropout to 0.7 for ett datasets
        # change head dropout to 0.7 for ett datasets
        if "ett" in DATASET:
            finetune_forecast_model = get_model(
                hf_model_path, context_length=context_length, prediction_length=forecast_length, head_dropout=0.7
            )
        else:
            finetune_forecast_model = get_model(
                hf_model_path, context_length=context_length, prediction_length=forecast_length
            )

        if freeze_backbone:
            print(
                "Number of params before freezing backbone",
                count_parameters(finetune_forecast_model),
            )

            # Freeze the backbone of the model
            for param in finetune_forecast_model.backbone.parameters():
                param.requires_grad = False

            # Count params
            print(
                "Number of params after freezing the backbone",
                count_parameters(finetune_forecast_model),
            )

        if learning_rate is None:
            learning_rate, finetune_forecast_model = optimal_lr_finder(
                finetune_forecast_model,
                dset_train,
                batch_size=BATCH_SIZE,
            )
            print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

        print(f"Using learning rate = {learning_rate}")
        finetune_forecast_args = TrainingArguments(
            output_dir=f"{SUBDIR}/fewshot_{fewshot_percent}",
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            num_train_epochs=EPOCHS,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            dataloader_num_workers=NUM_WORKERS,
            report_to=None,
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            logging_dir=f"{SUBDIR}/fewshot_{fewshot_percent}",  # Make sure to specify a logging directory
            load_best_model_at_end=True,  # Load the best model when training ends
            metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
            greater_is_better=False,  # For loss
            seed=SEED,
        )

        # Create the early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
            early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
        )
        tracking_callback = TrackingCallback()

        # Optimizer and scheduler
        optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            learning_rate,
            epochs=EPOCHS,
            steps_per_epoch=math.ceil(len(dset_train) / (BATCH_SIZE)),
        )

        finetune_forecast_trainer = Trainer(
            model=finetune_forecast_model,
            args=finetune_forecast_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            callbacks=[early_stopping_callback, tracking_callback],
            optimizers=(optimizer, scheduler),
        )
        finetune_forecast_trainer.remove_callback(INTEGRATION_TO_CALLBACK["codecarbon"])

        # Fine tune
        finetune_forecast_trainer.train()

        # Evaluation
        print(
            "+" * 20,
            f"Test MSE after few-shot {fewshot_percent}% fine-tuning",
            "+" * 20,
        )
        fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
        print(fewshot_output)
        print("+" * 60)

        # Plot
        plot_predictions(
            model=finetune_forecast_trainer.model,
            dset=dset_test,
            plot_dir=SUBDIR,
            num_plots=10,
            plot_prefix=f"test_fewshot_{fewshot_percent}",
            channel=0,
        )
        plt.close()

        # write results
        all_results[f"fs{fewshot_percent}_mse"].append(fewshot_output["eval_loss"])
        all_results[f"fs{fewshot_percent}_mean_epoch_time"].append(tracking_callback.mean_epoch_time)
        all_results[f"fs{fewshot_percent}_total_train_time"].append(tracking_callback.total_train_time)
        all_results[f"fs{fewshot_percent}_best_val_metric"].append(tracking_callback.best_eval_metric)

    df_out = pd.DataFrame(all_results).round(3)
    print(df_out[["dataset", "zs_mse", "fs5_mse"]])
    df_out.to_csv(f"{OUT_DIR}/results_zero_few.csv")
    df_out.to_csv(f"{OUT_DIR}/results_zero_few.csv")


# ## Benchmarking results*
# 
# *Some slight differences in the results as compared to the TTM paper results is possible due to different training environments.

# In[6]:


df_out


# In[ ]:




