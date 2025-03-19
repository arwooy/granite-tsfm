#!/usr/bin/env python
# coding: utf-8

# # Getting started with TinyTimeMixer (TTM) Rolling Predictions
# 
# This notebooke demonstrates the usage of a pre-trained or finetuned `TinyTimeMixer` model for rolling predictions. 
# 
# In this example, we will use a pre-trained TTM-512-96 model. That means the TTM model can take an input of 512 time points (`context_length`), and can forecast upto 96 time points (`forecast_length`) in the future. We then do rolling predictions of this model to keep predicting for longer forecast lengths.
# 
# Pre-trained TTM models will be fetched from the [Hugging Face TTM Model Repository](https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1). We also fine-tune a model and use the finetuned model for the same rolling predictions.

# In[1]:


import os
import tempfile

from transformers import Trainer, TrainingArguments, set_seed

from tsfm_public import TinyTimeMixerForPrediction, load_dataset
from tsfm_public.toolkit import RecursivePredictor, RecursivePredictorConfig
from tsfm_public.toolkit.visualization import plot_predictions


# ### Important arguments

# In[2]:


# Set seed for reproducibility
SEED = 42
set_seed(SEED)

# DATA ROOT PATH
# Make sure to download the target data (here ettm2) on the `DATA_ROOT_PATH` folder.
# ETT is available at: https://github.com/zhouhaoyi/ETDataset/tree/main
target_dataset = "etth1"

dataset_path = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

# DATA_ROOT_PATH = "/dccstor/tsfm23/datasets/"

# Results dir
OUT_DIR = "ttm_finetuned_models/"

# TTM model branch
# Use main for 512-96 model
# Use "1024_96_v1" for 1024-96 model
TTM_MODEL_REVISION = "main"

ROLLING_PREDICTION_LENGTH = 192
TTM_MODEL_URL = (
    "ibm-granite/granite-timeseries-ttm-v1"  # POINT TO A ZEROSHOT MODEL OR A FINETUNED MODEL TO DO ROLLING INFERENCE.
)


# ## LOAD BASE MODEL

# In[3]:


base_model = TinyTimeMixerForPrediction.from_pretrained(TTM_MODEL_URL, revision=TTM_MODEL_REVISION)

base_model_context_length = base_model.config.context_length
base_model_prediction_length = base_model.config.prediction_length

print(base_model_context_length, base_model_prediction_length)


# ## LOAD DATA

# In[4]:


_, _, dset_test = load_dataset(
    dataset_name=target_dataset,
    context_length=base_model_context_length,
    forecast_length=ROLLING_PREDICTION_LENGTH,
    fewshot_fraction=1.0,
    dataset_path=dataset_path,
)


# ## ROLLING PREDICTIONS

# In[5]:


rec_config = RecursivePredictorConfig(
    model=base_model,
    requested_prediction_length=ROLLING_PREDICTION_LENGTH,
    model_prediction_length=base_model_prediction_length,
    loss=base_model.config.loss,
)
rolling_model = RecursivePredictor(rec_config)


# In[6]:


temp_dir = tempfile.mkdtemp()
# zeroshot_trainer
zeroshot_trainer = Trainer(
    model=rolling_model,
    args=TrainingArguments(
        output_dir=temp_dir,
        per_device_eval_batch_size=32,
        seed=SEED,
    ),
)
# evaluate = zero-shot performance
print("+" * 20, "Test MSE zero-shot", "+" * 20)
zeroshot_output = zeroshot_trainer.evaluate(dset_test)
print(zeroshot_output)


# In[7]:


plot_predictions(
    model=zeroshot_trainer.model,
    dset=dset_test,
    plot_dir=os.path.join(OUT_DIR, target_dataset),
    plot_prefix="test_rolling",
    indices=[685, 118, 902, 1984, 894, 967, 304, 57, 265, 1015],
    channel=0,
)


# In[ ]:




