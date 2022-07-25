# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:34:06 2022

@author: aaron
"""

from sklearn.model_selection import train_test_split

from heart_attack_module import ExploratoryDataAnalysis,ModelDevelopment,ModelEvaluation
import pandas as pd
import numpy as np
import os

#%% Constants

DATA_PATH = os.path.join(os.getcwd(),'dataset','heart.csv')

#%% Exploratory Data Analysis

#%% 1) Data Loading

df = pd.read_csv(DATA_PATH)

#%% 2) Data Inspection

eda = ExploratoryDataAnalysis() 

eda.data_inspection(df)

# Defining categorical and continuous features

cat = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']
con = list(df.drop(cat,1))

df['thall'].replace(0,np.nan,inplace=True)
df['caa'].replace(4,np.nan,inplace=True)

eda.countplot(cat, df)
eda.distplot(con, df)

df.boxplot()

#%% 3) Data Cleaning

# Removing outliers - all features are within acceptable ranges

# Removing NaNs

# Using KNN Imputer

df_knn = eda.knn_imputer(df, cat)

df_knn.isna().sum()

# Removing duplicates

df_knn.duplicated().sum()
df_knn.drop_duplicates(inplace=True)
df_knn.duplicated().sum()

#%% 4) Features Selection

target = 'output'
selected_features = []

# Selecting continuous features using Logistic Regression
selected_features = eda.cat_vs_con_features_selection(df,con,target,
                                                      selected_features,0)

# Selecting categorical features using Cramer's V
selected_features = eda.cat_vs_cat_features_selection(df,cat,target,
                                                      selected_features,0)

selected_features.remove(target)

# Selected features are ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak', 'sex',
# 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']

# We can remove sex, fbs, restecg, exng, slp, and caa,
# but to test the test case given, we will include all features in this model

#%% 5) Data Preprocessing

X = df_knn[selected_features]
y = df_knn[target]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                 random_state=123)

#%% Model Development

md = ModelDevelopment() 

best_pipeline = md.ml_pipeline_classification(X_train, X_test, y_train, y_test)

# Best pipeline is MinMaxScaler with SVC

#%% Model Evaluation

me = ModelEvaluation()
cr = me.classification_report(X_test, y_test, best_pipeline, 'ml')

#%% Hyperparameter Tuning

param_grid = {'LRC__C':list(np.arange(1.0,2.0,0.2)),
               'LRC__solver':['lbfgs','newton-cg','liblinear','sag','saga'],
               'LRC__intercept_scaling':[1,2,3]}

grid = me.ml_grid_search(best_pipeline, param_grid, X_train, y_train)

print(grid.best_estimator_)
print(grid.best_score_)
print(grid.best_params_)




