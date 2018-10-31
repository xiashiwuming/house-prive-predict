#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 08:42:12 2018

@author: wuming
"""

import pandas as pd
import xgboost as xgb
from sklearn.metrics  import  accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
train=pd.read_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/house-price-predict/train.csv')
test=pd.read_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/house-price-predict/test.csv')
y_train=target=train['SalePrice'].values
train.drop('SalePrice',axis=1,inplace=True)
b=test['Id'].values


test_df=test.drop('Id',axis=1)
field_df = train.iloc[:,1:]

field_df.dropna(axis=1,thresh=780,inplace=True)
mmm=field_df.columns.values.tolist()
test_df=test_df[mmm]


object_list=field_df.select_dtypes(include='object').columns.values.tolist()
number_float64_list=field_df.select_dtypes(include='float64').columns.values.tolist()
number_int64_list=field_df.select_dtypes(include='int64').columns.values.tolist()

object_list1=test_df.select_dtypes(include='object').columns.values.tolist()
number_float64_list1=test_df.select_dtypes(include='float64').columns.values.tolist()
number_int64_list1=test_df.select_dtypes(include='int64').columns.values.tolist()



for i in range(len(object_list)):
    if field_df[object_list[i]].isnull().sum()!=1460:
        field_df.fillna(field_df.mode()[object_list[i]],inplace=True)
for i in range(len(number_float64_list)):
    if field_df[number_float64_list[i]].isnull().sum()!=1460:
        field_df[number_float64_list[i]].fillna(field_df[number_float64_list[i]].mean(),inplace=True)        
for i in range(len(number_int64_list)):
    if field_df[number_int64_list[i]].isnull().sum()!=1460:
        field_df[number_int64_list[i]].fillna(field_df[number_int64_list[i]].mean(),inplace=True)        


##测试集合的操做：
for i in range(len(number_float64_list1)):
    if test_df[number_float64_list1[i]].isnull().sum()!=1460:
        test_df[number_float64_list1[i]].fillna(test_df[number_float64_list1[i]].mean(),inplace=True)        
for i in range(len(number_int64_list)):
    if test_df[number_int64_list[i]].isnull().sum()!=1460:
        test_df[number_int64_list[i]].fillna(test_df[number_int64_list[i]].mean(),inplace=True)        
        
        
        







for i in range(len(object_list)):
    a={}
    for k in range(len(field_df[object_list[i]].unique())):
       a[field_df[object_list[i]].unique()[k]]=k
    field_df[object_list[i]]=field_df[object_list[i]].map(a) 
    
    
for i in range(len(object_list1)):
    if test_df[object_list1[i]].isnull().sum()!=1459:
        test_df[object_list1[i]].fillna(test_df[object_list1[i]].mode()[0])
for i in range(len(object_list1)):
    a={}
    for k in range(len(test_df[object_list1[i]].unique())):
       a[test_df[object_list1[i]].unique()[k]]=k
    test_df[object_list1[i]]=test_df[object_list1[i]].map(a)       
    
    
    
            
X_train=field_df.values
test_df1=test_df.values
cv_params = {'n_estimators': [400, 500, 600, 700, 800,900,1000]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**other_params)
model=optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
model.fit(X_train, y_train)
a=model.predict(test_df1)
datafram=pd.DataFrame({'Id':b,'SalePrice':a})
datafram.to_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/house-price-predict/submit10.csv',index=False,sep=',')

