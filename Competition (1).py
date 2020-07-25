#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import mean_absolute_error

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV


# In[10]:


dlt = pd.read_csv("dengue_labels_train.csv")
dft = pd.read_csv("dengue_features_train.csv")
tdf = pd.read_csv("dengue_features_test.csv")

df = dlt.merge(dft)
y = df["total_cases"]
df = df.drop("total_cases", axis=1)
df = pd.concat([df, tdf])


df = df.drop("week_start_date", axis=1)
df = df.fillna(df.mean())
df = pd.get_dummies(df, columns = ["city", "year", "weekofyear"])




'''
df = df.fillna(df.mean())
df = df.drop("week_start_date", axis=1)
y = df["total_cases"]
X = df.drop("total_cases", axis=1)
X = pd.get_dummies(X, columns = ["city", "year", "weekofyear"])
'''
'''
tdf = tdf.fillna(df.mean())
tdf = tdf.drop("week_start_date", axis=1)
X_test = pd.get_dummies(tdf, columns = ["city", "year", "weekofyear"])
'''

from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()
scaler1.fit(df)
df = scaler1.transform(df)
trainset = pd.DataFrame(df).head(1456)
testset  = pd.DataFrame(df).tail(len(df) - len(trainset))


print(trainset.shape)
print(testset.shape)
testset = testset.set_index( pd.Index( list( range( 0,len(testset) ) ) ) )
trainset = trainset.set_index( pd.Index( list( range( 0,len(trainset) ) ) ) )
#print(trainset.head())
#print(testset.head())
'''
scaler2 = StandardScaler()
scaler2.fit(X_test)
X_test = scaler2.transform(X_test)
'''

from sklearn.feature_selection import f_regression

pvalues = f_regression(trainset,y)
array = list(pvalues[1])

def removeHighPValueFeature(array,trainset,testset,threshold):
    col = 0
    for v in array:
        if v > threshold:
            col = int(array.index(v,col))
            try:
                trainset = trainset.drop(col, axis=1)
                testset = testset.drop(col, axis=1)
            except:
                col += 1
                trainset = trainset.drop(col, axis=1)
                testset = testset.drop(col, axis=1)
    return trainset,testset

#X = pd.DataFrame(X)
#X_test = pd.DataFrame(X_test)

X,X_test = removeHighPValueFeature(array,trainset,testset,0.01)
#X = trainset
#X_test = testset


# In[12]:


reg1 = SVR().set_params(**{'C': 30, 'epsilon': 1.0})
reg2 = MLPRegressor().set_params(**{'activation': 'relu', 'hidden_layer_sizes': (100, 30), 'solver': 'sgd'})
reg3 = BaggingRegressor().set_params(**{'n_estimators': 50})
estimators=[('svr', reg1), ('mlp', reg2), ('bgr', reg3)]

from sklearn.ensemble import VotingRegressor
vereg = VotingRegressor(estimators=estimators)

vereg.fit(X, y)
y_pred = vereg.predict(X_test)
y_pred = [int(round(abs(elem))) for elem in y_pred]

#grad = GradientBoostingRegressor()
#grad.fit(X, y)
#y_pred = grad.predict(X_test)
#y_pred = [int(abs(elem)) for elem in y_pred]

#os.chdir(r'C:\Users\Mustafa Kaan\Desktop')
pd.DataFrame(y_pred).to_csv("comp4.csv")


# In[ ]:




