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
del data_features['scheme_name']
del data_features['recorded_by']
del data_features["water_quality"]
del data_features['quantity_group']
del data_features["ward"]
del data_features["num_private"]
del data_features['scheme_management']
del data_features["extraction_type"]
del data_features["extraction_type_class"]
del data_features["region_code"]
del data_features["waterpoint_type_group"]
del data_features["source_type"]
del data_features["subvillage"]
del data_features["management_group"]

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

##########################################################
#look into merging most of the extraction type classses into one category 
pd.set_option("display.max_columns",200)
print(pd.crosstab(data_features["extraction_type_class"], data_features["extraction_type"],margins = False))

c = pd.crosstab(data_features.extraction_type_class, data_features.extraction_type).stack().reset_index(name='C')
c.plot.scatter('extraction_type_class', 'extraction_type', s=c.C * 0.1)

table = pd.crosstab(data_features["extraction_type_group"],data_features.extraction_type)
table = pd.crosstab(data_features["extraction_type_group"],data_features.extraction_type_class)


# Note, extraction type is a subset of extraction type group and that is a subset of ectraction type class
# Let's move for now with extraction_type group as it includes information on specific handpumps (still up to 2000 wells per handpump)

##########################################################

#plotting the map with wells
print(data_features["longitude"].isin([0]).sum())
missing_longitudes = data_features[ data_features['longitude'] == 0 ]
mapping= data_features.loc[:,['longitude','latitude']]
mapping = mapping.drop(missing_longitudes.index, axis=0)
plt.scatter(x=mapping['longitude'], y=mapping['latitude'])
plt.show()
        
x1=countdataocc(data_features["water_quality"])
x2=countdataocc(data_features["quality_group"])

#water quality, quality group 

c = pd.crosstab(data_features.water_quality, data_features.quality_group).stack().reset_index(name='C')
c.plot.scatter('water_quality', 'quality_group', s=c.C * 0.1)

##################lga,ward #########################################l
#basically regional governmental areas (ward, local government area)

data_features.ward.value_counts()
data_features.ward.head()

data_features.lga.value_counts()
data_features.lga.head()

##########################################################

#Waterpoint name(wpt_name)
data_features["wpt_name"].value_counts()
data_features["wpt_name"].value_counts().head(10)

most_common_names = data_features["wpt_name"].value_counts()[0:5].index
for name in most_common_names:
    
    subset = data_features[data_features["wpt_name"]== name]
    print(name + " "+str(len(subset)))
    print(subset.label.value_counts()/len(subset))
    
data_features.loc[~data_features.wpt_name.isin(most_common_names),"wpt_name"] = "other"

#########################Waterpoint Type and Waterpoint type group#################################

table = pd.crosstab(data_features.waterpoint_type,data_features.waterpoint_type_group)

# They are nearly the same. The only difference is that waterpoint type make a difference 
#between to standpipes. With at least 6000 entries each this difference might be reasonable
#Let's go with waterpoint_type for now


###############################Management group##############
# lets delete it for now as 50000 are the same ("user group") maybe later we can reintroduce it
data_features.management_group.value_counts()
data_features.groupby("management_group").label.value_counts()


##########################################################
         
# NUm_private can be deleted entirely
Num_private_value_rows = data_features[data_features.num_private != 0]
plt.hist(Num_private_value_rows.label)
Num_private_value_rows.label.value_counts()/len(Num_private_value_rows)

##########################################################


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

###############################Region & Region Code###########################

table = pd.crosstab(data_features["region"],data_features["region_code"])

# examening and plotting different regions and codes:
    
Arusha_region = data_features[data_features["region"]== "Arusha"]
x24 = Arusha_region["longitude"][Arusha_region["region_code"]==24]
y24 = Arusha_region["latitude"][Arusha_region["region_code"]==24]
x2 = Arusha_region["longitude"][Arusha_region["region_code"]==2]
y2 = Arusha_region["latitude"][Arusha_region["region_code"]==2]
plt.plot(x24,y24,"o",color = "red")
plt.plot(x2,y2,"o",color = "blue")

Mtwara_region = data_features[data_features["region"]== "Mtwara"]
x90 = Mtwara_region["longitude"][Mtwara_region["region_code"]==90]
y90 = Mtwara_region["latitude"][Mtwara_region["region_code"]==90]
x99 = Mtwara_region["longitude"][Mtwara_region["region_code"]==99]
y99= Mtwara_region["latitude"][Mtwara_region["region_code"]==99]
x9 = Mtwara_region["longitude"][Mtwara_region["region_code"]==9]
y9 = Mtwara_region["latitude"][Mtwara_region["region_code"]==9]
plt.plot(x90,y90,"o",color = "red")
plt.plot(x9,y9,"o",color = "green")
plt.plot(x99,y99,"o",color = "blue")

# Let's go with region first, the columns are nearly the same and in the Mtwara region region_code seems to make less sense




##########################################################




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

######################################Source & Source Type #########################################################
data_features.source.value_counts()
data_features.source_type.value_counts()

contingency_table = pd.crosstab(data_features.source,data_features.source_type)
#source type can be deleted; 

######################################Source & Source Class #########################################################
data_features.source_class.value_counts()
data_features.source.value_counts()

contingency_table = pd.crosstab(data_features.source,data_features.source_class)

##########################################################region & district code######
data_features.region.value_counts()
data_features.district_code.value_counts()

table = pd.crosstab(data_features.region,data_features.district_code)

# nothing clear I can find from here

##########################################################region & lga######


table = pd.crosstab(data_features.region,data_features.lga)
# seems like regions are made up of lgas

######################################Cleaning#########################################################



######################################Funder & Installer#########################################################

data_features.installer[data_features.installer.isnull()] = "Missing"
data_features.installer[data_features.installer == "0"] = "Missing"
most_common_installer = data_features.installer.value_counts().index[data_features.installer.value_counts() > 400]
data_features.installer[~data_features.installer.isin(most_common_installer)] = "other"
x = data_features.installer.value_counts()

data_features.funder[data_features.funder.isnull()] = "Missing"
data_features.funder[data_features.funder == "0"] = "Missing"
most_common_funder = data_features.funder.value_counts().index[data_features.funder.value_counts() > 400]
data_features.funder[~data_features.funder.isin(most_common_funder)] = "other"
x = data_features.funder.value_counts()

contingency_table = pd.crosstab(data_features.installer,data_features.funder)

#some just seem to be missspelled or spelled differently
#Commu and Community might be the same and amybe we can combine a lot of the government stuff 

data_features.installer[data_features.installer == "HESAWA"] = "hesawa"
data_features.installer[data_features.installer == "Hesawa"] = "hesawa"
data_features.installer[data_features.installer == "DANID"] = "DANIDA"



######################################Prediction Part#########################################################


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
#Note, baseline accuracy should probably be 54% as that is the accuracy you get if you always choose 0 

reversefactor = dict(zip(range(3),definitions))
rf_validation_data["label"] = np.vectorize(reversefactor.get)(rf_validation_data["label"])
prediction = np.vectorize(reversefactor.get)(prediction)

contingency_table = pd.crosstab(rf_validation_data["label"], prediction, rownames=['Actual Label'], colnames=['Predicted Label'])
print(contingency_table)
overall_accuracy = accuracy_score(rf_validation_data["label"],prediction)
print(overall_accuracy)

