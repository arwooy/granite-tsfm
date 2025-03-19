#!/usr/bin/env python
# coding: utf-8

# # Install IBM-Granite/granite-tsfm repository
# 
# This notebook installs the IBM Time Series Foundation Model repository.

# In[ ]:


# Install ibm/tsfm
get_ipython().system(' pip install "tsfm_public[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.12"')


# In[ ]:


# Check installation
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction


model_512 = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-v1", revision="main")
model_512.config.context_length


# In[ ]:


# Check for another tsfm model
model_1024 = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-v1", revision="1024_96_v1")
model_1024.config.num_patches

