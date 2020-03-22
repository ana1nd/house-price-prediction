# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:27:22 2020

@author: annamdev
"""

import pandas as pd

trainFileName = "train.csv"
testFileName = "test.csv"

def GetFiles():
    
    train = pd.read_csv(trainFileName)
    test = pd.read_csv(testFileName)
    
    return train, test
    
def ExploratoryDataAnalysis():
    
    # for each feature count nulls/NA/0
    # check for outliers - remove them