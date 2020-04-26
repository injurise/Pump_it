#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:42:04 2020

@author: alexandermollers
"""

import pandas as pd
import matplotlib.pyplot as plt

data_features = pd.read_csv("Pump_it_Up_training_features.csv")
data_lables = pd.read_csv("Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_labels.csv")

data_features.hist(column=["amount_tsh"],bins=1000)
plt.xlim(0,6000)

# seems to be one kinda outlier with a lot of water:
print(data_features["amount_tsh"].max())
print(data_features.index[data_features["amount_tsh"]==data_features["amount_tsh"].max()])
