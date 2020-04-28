#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:42:04 2020

@author: alexandermollers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_features = pd.read_csv("Pump_it_Up_training_features.csv", index_col = "id")
data_lables = pd.read_csv("Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_labels.csv", index_col = "id")

data_features.head()
data_features.tail()
list(data_features.columns)

plt.figure(figsize = (12,6))
plt.xlim(0,6000)
plt.hist(data_features["amount_tsh"],bins = 1000)

data_features["region_code"].nunique()
data_features["region"].nunique()
#5 less regions than region codes? 


sns.barplot(x = data_features["region_code"], y = data_features["population"])
plt.figure(figsize = (12,6))
# scatterplot isn't so useful for the categorical data I guess, probably swarmplot and violinplot below are better
sns.scatterplot(x = data_features["water_quality"], y = data_features["population"], hue = data_features["quantity"])
plt.figure(figsize = (12,6))

#Don't uncomment the swarmplot, it only really works for less data
#Try the ciolinplot below instead
#sns.swarmplot(x = data_features["water_quality"], y = data_features["population"])

sns.violinplot(x = data_features["water_quality"], y = data_features["population"])

#sns.heatmap(data_features[["basin","population"]])


# seems to be one kinda outlier with a lot of water (around 20000):
print(data_features["amount_tsh"].max())
print(data_features.index[data_features["amount_tsh"]==data_features["amount_tsh"].max()])
