#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:42:04 2020

@author: alexandermollers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def countdataocc(column): 
    dic= {}
    for entry in column:
        if entry in dic:
            dic[entry] += 1
        else:
            dic[entry] = 1
    
    return dic
x1=countdataocc(data_features["water_quality"])
x2=countdataocc(data_features["quality_group"])

data_features = pd.read_csv("Training_set_features.csv", index_col = "id")
data_lables = pd.read_csv("Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_labels.csv", index_col = "id")

data_features.head()
data_features.tail()

data_features.describe()

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



sns.scatterplot(x = data_features["water_quality"], y = data_features["quality_group"], hue = data_features["quantity"])
plt.figure(figsize = (12,6))

#Checking if variables are the same
pd.set_option("display.max_columns",200)
print(pd.crosstab(data_features["extraction_type_class"], data_features["extraction_type"],margins = False))

c = pd.crosstab(data_features.source, data_features.source_type).stack().reset_index(name='C')
c.plot.scatter('source', 'source_type', s=c.C * 0.1)


#Inspecting distribution of one specific category of the feature
x = data_features[data_features["extraction_type_group"] == "swn 80"] 
x.plot.scatter(x, 'extraction_type_group', s=c.C * 0.1)

#replacing unknown to nan
data_features.replace(to_replace="unknown",value="nan", inplace=True)
data_features.replace(to_replace="Not Known",value="nan", inplace=True)
#checking for missing categorical data
print(data_features.isnull().sum())

#checking the total num of 0s in a colums
print(data_features["gps_heigth"].isin([0]).sum())

#Deleting duplicate columns
del data_features['payment']

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
            

