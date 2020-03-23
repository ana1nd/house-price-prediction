# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:27:22 2020

@author: annamdev
"""

import pandas as pd
import numpy as np
import seaborn as sns

trainFileName = "train.csv"
testFileName = "test.csv"

def GetFiles():
    
    train = pd.read_csv(trainFileName)
    test = pd.read_csv(testFileName)
    
    return train, test
    
def FindCorrelationWithEachFeature(data, features, targetFeature):
    
    featureCorrTupleList = []
    for feature in features:
        if (feature == targetFeature) :
            continue # to avoid 1.0 corr with same target feature 
        correlation = data[feature].corr(data[targetFeature])
        featureCorrTupleList.append((feature, correlation))
        
    featureCorrTupleList.sort(key= lambda x: x[1])
    featureCorrTupleList.reverse()
    return featureCorrTupleList
    
def FindPositiveCorrFeatures(features):
        
    threshold = 0 # you can keep varying this threshold
    goodFeatures = []
    for i in range(0, len(features), 1):
        if(features[i][1]< threshold):
            return goodFeatures
        goodFeatures.append(features[i][0])
    return goodFeatures
    
def ExploratoryDataAnalysis(train):
    
    # for each feature count nulls/NA/0
    # check for outliers - remove them
    
    # separate numerical and cateogorical data
    numeric = train.select_dtypes(include = [np.number])
    
    # find corr of each feature with target variable
    numericCorrList = FindCorrelationWithEachFeature(numeric, numeric.columns.tolist(), 'SalePrice')
    
    # find positive correlation features
    numericGoodFeatures = FindPositiveCorrFeatures(numericCorrList)
    numeric = train[numericGoodFeatures]
    
    # find outliers (plot and either remove(if less outliers) those/or change those with mean values)
    
    # find category features    
    category = train.select_dtypes(exclude = [np.number])
    category.loc[:, 'SalePrice'] = train['SalePrice'] # add target feature since it's not category feature
    
    # do label encoding, one-hot encoding for category features
    
    # find correlation for each category feature with targetFeature value
    categoryCorrList = FindCorrelationWithEachFeature(category, category.columns.tolist(), 'SalePrice')
    
    # outlier detection/modification for category features