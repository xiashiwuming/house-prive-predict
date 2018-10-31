#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:30:34 2018

@author: wuming
"""
import pandas as pd
import xgboost as xgb
from sklearn.metrics  import  accuracy_score
import numpy as np
train=pd.read_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/boston-housing/train.csv')
test=pd.read_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/boston-housing/test.csv')
y_train=target=train['medv'].values
train.drop('medv',axis=1,inplace=True)
b=test['ID'].values


test_df=test.drop('ID',axis=1)
field_df = train.iloc[:,1:]


X_train=field_df1=field_df.values
test_df1=test_df.values
cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**other_params)
model=optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
model.fit(X_train, y_train)
a=model.predict(test_df1)
datafram=pd.DataFrame({'ID':b,'medv':a})
datafram.to_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/boston-housing/submit3.csv',index=False,sep=',')

