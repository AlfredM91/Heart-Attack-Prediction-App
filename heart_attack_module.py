# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:34:36 2022

@author: aaron
"""


from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

#%% Constants

BEST_MODEL_PATH = os.path.join(os.getcwd(),'models','best_model.h5')
BEST_PIPELINE_PATH = os.path.join(os.getcwd(),'models','model.pkl')
GRID_PATH = os.path.join(os.getcwd(),'models','grid_best_estimator.pkl')

#%% Classes

class ExploratoryDataAnalysis:
    def data_inspection(self,df):
        pd.set_option('display.max_columns',None)
        print(df.describe(include='all').T)
        print(df.info())
        print(df.isna().sum())
        print(df.isnull().sum())
        print(df.duplicated().sum())
    
    def countplot(self,cat,df):
        for i in cat:
            plt.figure()
            sns.countplot(df[i])
            plt.show()

    def distplot(self,con,df):
        for i in con:
            plt.figure()
            sns.distplot(df[i])
            plt.show()
    
    def lineplot(self,con,df):
        for i in con:
            plt.figure()
            plt.plot(df[i])
            plt.show()
            
    def knn_imputer(self,df,cat):
        knn = KNNImputer() 
        df_knn = df
        df_knn = knn.fit_transform(df)
        df_knn = pd.DataFrame(df_knn)
        df_knn.columns = df.columns
        for i in cat:
            df_knn[i] = np.floor(df_knn[i]).astype(int)
        print(df_knn.describe().T)
        return df_knn
    
    def cat_vs_con_features_selection(self,df,con,target,selected_features,
                                      target_score=0.6,
                                      solver='lbfgs',max_iter=100):
        
        for i in con:
            lr = LogisticRegression(solver=solver,max_iter=max_iter)   
            lr.fit(np.expand_dims(df[i],axis=-1),df[target])
            lr_score = lr.score(np.expand_dims(df[i],axis=-1),df[target])
            print(i)
            print(lr_score)
            if lr_score >= target_score:
                selected_features.append(i)
        return selected_features
    
    def cat_vs_cat_features_selection(self,df,cat,target,selected_features,
                                      target_score=0.6):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        for i in cat:
            cramers_confusion_matrix = pd.crosstab(df[i],df[target]).to_numpy()
            chi2 = ss.chi2_contingency(cramers_confusion_matrix)[0]
            n = cramers_confusion_matrix.sum()
            phi2 = chi2/n
            r,k = cramers_confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            cramers_score = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
            print(i)
            print(cramers_score)
            if cramers_score >= target_score:
                selected_features.append(i)
                # print(i,' ',cramers_score)
        return selected_features

class ModelDevelopment:
    def ml_pipeline_classification(self,X_train,X_test,y_train,y_test):
        
        # Logistic Regression
        pipeline_mms_lr = Pipeline([('MMS',MinMaxScaler()),
                                    ('LRC',LogisticRegression(random_state=123))])

        pipeline_ss_lr = Pipeline([('SS',StandardScaler()),
                                    ('LRC',LogisticRegression(random_state=123))])

        # Decision Tree
        pipeline_mms_dt = Pipeline([('MMS',MinMaxScaler()),
                                    ('RTC',DecisionTreeClassifier(random_state=123))])

        pipeline_ss_dt = Pipeline([('SS',StandardScaler()),
                                    ('RTC',DecisionTreeClassifier(random_state=123))])

        # Random Forest
        pipeline_mms_rf = Pipeline([('MMS',MinMaxScaler()),
                                    ('RFC',RandomForestClassifier(random_state=123))])

        pipeline_ss_rf = Pipeline([('SS',StandardScaler()),
                                    ('RFC',RandomForestClassifier(random_state=123))])

        # SVC
        pipeline_mms_svm = Pipeline([('MMS',MinMaxScaler()),
                                    ('SVC',SVC(random_state=123))])

        pipeline_ss_svm = Pipeline([('SS',StandardScaler()),
                                    ('SVC',SVC(random_state=123))])

        # KNN
        pipeline_mms_knn = Pipeline([('MMS',MinMaxScaler()),
                                    ('KNC',KNeighborsClassifier())])

        pipeline_ss_knn = Pipeline([('SS',StandardScaler()),
                                    ('KNC',KNeighborsClassifier())])

        # GBoost
        pipeline_mms_gb = Pipeline([('MMS',MinMaxScaler()),
                                    ('GBC',GradientBoostingClassifier(random_state=123))])

        pipeline_ss_gb = Pipeline([('SS',StandardScaler()),
                                    ('GBC',GradientBoostingClassifier(random_state=123))])

        # create a list to store all the pipelines

        pipelines = [pipeline_mms_lr, pipeline_ss_lr,
                     pipeline_mms_dt,pipeline_ss_dt,
                     pipeline_mms_rf,pipeline_ss_rf,
                     pipeline_mms_svm,pipeline_ss_svm,
                     pipeline_mms_knn,pipeline_ss_knn,
                     pipeline_mms_gb,pipeline_ss_gb]

        for pipe in pipelines:
            pipe.fit(X_train,y_train)

        best_accuracy = 0

        for i, pipe in enumerate(pipelines):
            print(i," ",pipe.steps," ",pipe.score(X_test,y_test))
            if pipe.score(X_test,y_test) > best_accuracy:
                best_accuracy = pipe.score(X_test,y_test)
                best_pipeline = pipe
                
        print('The best scaler and classifier is {} with accuracy of {}'.
              format(best_pipeline.steps,best_accuracy))
        
        with open(BEST_PIPELINE_PATH,'wb') as file:
            pickle.dump(best_pipeline,file)
        
        return best_pipeline

class ModelEvaluation:
    def ml_grid_search(self,best_pipeline,param_grid,X_train,y_train,
                       scoring=None):
        grid_search = GridSearchCV(estimator=best_pipeline,
                                   param_grid=param_grid,
                                   scoring=scoring,
                                   cv=5,
                                   verbose=1,
                                   n_jobs=-1)
        grid = grid_search.fit(X_train,y_train)
        print('Best Score = {}'.format(grid.best_score_))
        print(grid.best_params_)

        with open(GRID_PATH,'wb') as file:
            pickle.dump(grid.best_estimator_,file)
        return grid

    def classification_report(self,X_test,y_test,best_model,ml_or_dl):
                    
        if ml_or_dl=='ml':
            y_pred = best_model.predict(X_test)
            y_true = y_test
        elif ml_or_dl=='dl':
            y_pred = best_model.predict(X_test)
            y_pred = np.argmax(y_pred,axis=1)
            y_true = np.argmax(y_test,axis=1)
        else:
            print('Please put either ''ml'' or ''dl'' for the ml_or_dl argument')

        cr = classification_report(y_true, y_pred)
        print(cr)
        return cr
    