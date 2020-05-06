#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:42:04 2020

@author: alexandermollers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def countdataocc(column): 
    dic= {}
    for entry in column:
        if entry in dic:
            dic[entry] += 1
        else:
            dic[entry] = 1
    
    return dic



data_features = pd.read_csv("Training_set_features.csv", index_col = "id")
data_labels = pd.read_csv("Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_labels.csv", index_col = "id")

data_features.head()
data_features.tail()

data_features.describe()

#check if data feature Ids are the same as 
print(np.all(data_features.index == data_labels.index))
#Add labels to trainings_df for analysis
data_features["label"]=data_labels["status_group"]


#Deleting duplicate columns
del data_features['payment']

overview = data_features["funder"].value_counts()
percentage = overview/overview.sum()
percentage_working_by_funder = data_features.groupby(["funder","label"]).size().unstack()
percentage_working_by_funder["total"] = percentage_working_by_funder.sum(axis =1)
percentage_working_by_funder.functional  = percentage_working_by_funder["functional"]/percentage_working_by_funder["total"]
percentage_working_by_funder["functional needs repair"]  = percentage_working_by_funder["functional needs repair"]/percentage_working_by_funder["total"]
percentage_working_by_funder["non functional"]  = percentage_working_by_funder["non functional"]/percentage_working_by_funder["total"]


#check how many occurences of each label we have
#two ways:
#simple hist with pd
data_features["label"].value_counts().plot.bar()
data_features.groupby(["label", 'water_quality']).size().unstack().plot.bar(stacked=True)

dftest=data_features[data_features["extraction_type_class"]=="handpump"]
dftest.groupby(["label", 'extraction_type']).size().unstack().plot.bar(stacked=True)
#stacked hist with matplot
plt.bar(data_features["label"].value_counts().index,data_features["label"].value_counts())


#look into merging most of the extraction type classses into one category 


        
x1=countdataocc(data_features["water_quality"])
x2=countdataocc(data_features["quality_group"])
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

c = pd.crosstab(data_features.extraction_type_class, data_features.extraction_type_group).stack().reset_index(name='C')
c.plot.scatter('extraction_type_class', 'extraction_type_group', s=c.C * 0.1)

#Inspecting distribution of one specific category of the feature
swn_rows = data_features[data_features["extraction_type_group"] == "swn 80"] 
plt.hist(swn_rows["label"])

#checking for missing data, although it doesn't count unknown and 0 values
print(data_features.isnull().sum())
#checking the total num of 0s in a colums
print(data_features["construction_year"].isin([0]).sum())

#replacing unknown to nan
data_features.replace(to_replace="unknown",value="nan", inplace=True)
data_features.replace(to_replace="Not Known",value="nan", inplace=True)
#checking for missing categorical data
print(data_features.isnull().sum())

#checking the total num of 0s in a colums
print(data_features["gps_heigth"].isin([0]).sum())




#Label encodedclasses and store definitions
label_encoded= pd.factorize(data_features["label"])
data_features.label = label_encoded[0]
definitions = label_encoded[1]

training_features = ["quantity_group","water_quality"]
random_forest_data = data_features[training_features]
random_forest_data = pd.get_dummies(random_forest_data)
random_forest_data["label"] = data_features.label
#Split data into training (0.8),validation (0.1) and test set (0.1)
#maybe we can use a stratified split later on, but for now this should do


rf_train_data = random_forest_data[:int(0.8*len(random_forest_data))]
rf_validation_data = random_forest_data[int(-0.2*len(random_forest_data)):-int(0.1*len(random_forest_data))]
rf_test_data = random_forest_data[int(-0.1*len(random_forest_data)):]

#we can check the distribution of the labels but the split seems okay
print(rf_train_data.label.value_counts()/len(rf_train_data))
print(rf_validation_data.label.value_counts()/len(rf_validation_data))
print(rf_test_data.label.value_counts()/len(rf_test_data))

#Fit randomforest and predict labels

rf = RandomForestClassifier(max_depth=10, random_state=0)
fit = rf.fit(rf_train_data.drop(columns = ["label"]),rf_train_data["label"])

prediction = fit.predict(rf_validation_data.drop(columns = ["label"]))

#evaluate accuracy of predictor

reversefactor = dict(zip(range(3),definitions))
rf_validation_data["label"] = np.vectorize(reversefactor.get)(rf_validation_data["label"])
prediction = np.vectorize(reversefactor.get)(prediction)

contingency_table = pd.crosstab(rf_validation_data["label"], prediction, rownames=['Actual Label'], colnames=['Predicted Label']))
print(contigency_tabel)
overall_accuracy = accuracy_score(rf_validation_data["label"],prediction)
print(overall_accuracy)

