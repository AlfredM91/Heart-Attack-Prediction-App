# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:34:36 2022

@author: aaron
"""

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, SimpleRNN, GRU, Masking
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

#%% Constants

BEST_MODEL_PATH = os.path.join(os.getcwd(),'models','best_model.h5')
OHE_PATH = os.path.join(os.getcwd(),'models','ohe.pkl')
MMS_PATH = os.path.join(os.getcwd(),'models','mms.pkl')
SS_PATH = os.path.join(os.getcwd(),'models','ss.pkl')
BEST_PIPELINE_PATH = os.path.join(os.getcwd(),'models','model.pkl')
PLOT_PATH = os.path.join(os.getcwd(),'statics','model.png')
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
MODEL_PATH = os.path.join(os.getcwd(),'models','model.h5')
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
            
    def label_encoder(self,cat_exclude_target,df):
        for i in cat_exclude_target:
            le = LabelEncoder()
            temp = df[i]
            temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()])
            df[i] = pd.to_numeric(temp,errors='coerce')
            ENCODER_PATH = os.path.join(os.getcwd(),'models',i + '_encoder.pkl')
            pickle.dump(le,open(ENCODER_PATH,'wb'))
        return df
    
    def one_hot_encoder(self,y):
        ohe = OneHotEncoder(sparse=False)
        if y.ndim == 1:
            y = ohe.fit_transform(np.expand_dims(y,axis=-1))
        else:
            y = ohe.fit_transform(y)

        with open(OHE_PATH,'wb') as file:
            pickle.dump(OHE_PATH,file)
        return y
    
    def simple_imputer(self,df,cat,con):
        
        df_simple = df
        
        for i in con:
            df_simple[i] = df[i].fillna(df[i].median())

        for i in cat:
            df_simple[i] = df[i].fillna(df[i].mode()[0])

        print(df_simple.describe().T)
        return df_simple

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
    
    def iterative_imputer(self,df,cat):
        ii = IterativeImputer() 
        df_ii = df
        df_ii = ii.fit_transform(df)
        df_ii = pd.DataFrame(df_ii)
        df_ii.columns = df.columns
        for i in cat:
            df_ii[i] = np.floor(df_ii[i]).astype(int)
        print(df_ii.describe().T)
        return df_ii
    
    def con_vs_con_features_selection(self,df,con,target,selected_features,
                                      corr_target=0.6,figsize=(20,12)):
        
        cor = df.loc[:,con].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(cor,cmap=plt.cm.Reds,annot=True)
        plt.show()
                
        for i in con:
            print(i)
            print(cor[i].loc[target])
            if (cor[i].loc[target]) >= corr_target:
                selected_features.append(i)
                print(i,' ',cor[i].loc[target])
        return selected_features
    
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
    
    def con_vs_cat_features_selection(self):
        return self
    
    def min_max_scaler(self,X):
        mms = MinMaxScaler()    
        
        if X.ndim == 1:
            X = mms.fit_transform(np.expand_dims(X,axis=-1))
        else:    
            X = mms.fit_transform(X)


        with open(MMS_PATH,'wb') as file:
            pickle.dump(mms,file)
        
        return X
    
    def standard_scaler(self,X):
        ss = StandardScaler()    
        
        if X.ndim == 1:
            X = ss.fit_transform(np.expand_dims(X,axis=-1))
        else:    
            X = ss.fit_transform(X)
            
        with open(SS_PATH,'wb') as file:
            pickle.dump(ss,file)
        
        return X

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

    def ml_pipeline_regression(self,X_train,X_test,y_train,y_test,
                               evaluation_metric=None):
        
        # Linear Regression
        pipeline_mms_lr = Pipeline([('Min_Max_Scaler',MinMaxScaler()),
                                    ('Linear_Classifier',
                                     LinearRegression())])

        pipeline_ss_lr = Pipeline([('Standard_Scaler',StandardScaler()),
                                    ('Linear_Classifier',
                                     LinearRegression())])
        
        # Lasso Regression
        pipeline_mms_lasso = Pipeline([('Min_Max_Scaler',MinMaxScaler()),
                                    ('Lasso',Lasso(random_state=123))])

        pipeline_ss_lasso = Pipeline([('Standard_Scaler',StandardScaler()),
                                    ('Lasso',Lasso(random_state=123))])
        
        # Ridge Regression
        pipeline_mms_ridge = Pipeline([('Min_Max_Scaler',MinMaxScaler()),
                                    ('Ridge',Ridge(random_state=123))])

        pipeline_ss_ridge = Pipeline([('Standard_Scaler',StandardScaler()),
                                    ('Ridge',Ridge(random_state=123))])

        # create a list to store all the pipelines

        pipelines = [pipeline_mms_lr, pipeline_ss_lr,
                     pipeline_mms_lasso,pipeline_ss_lasso,
                     pipeline_mms_ridge,pipeline_ss_ridge]

        for pipe in pipelines:
            pipe.fit(X_train,y_train)

        best_r2 = 0
        worst_error = float('inf')

        for i, pipe in enumerate(pipelines):
            y_pred = pipe.predict(X_test)
            y_true = y_test
            print(i," ",pipe.steps," ",pipe.score(X_test,y_test))
            print('MAE: {}'.format(mean_absolute_error(y_true,y_pred)))
            print('RMSE: {}'.format(mean_squared_error(y_true,y_pred,squared=False)))
            print('MAPE: {}'.format(mean_absolute_percentage_error(y_true,y_pred)))
            
            if evaluation_metric=='R2':
                if pipe.score(X_test,y_test) > best_r2:
                    best_evaluation_metric_value = pipe.score(X_test,y_test)
                    best_pipeline = pipe
            elif evaluation_metric=='MAE':
                if mean_absolute_error(y_true,y_pred) < worst_error:
                    best_evaluation_metric_value = mean_absolute_error(y_true,y_pred)
                    best_pipeline = pipe
            elif evaluation_metric=='RMSE':
                if mean_squared_error(y_true,y_pred,squared=False) < worst_error:
                    best_evaluation_metric_value = mean_squared_error(y_true,y_pred)
                    best_pipeline = pipe
            elif evaluation_metric=='MAPE':
                if mean_absolute_percentage_error(y_true,y_pred) < worst_error:
                    best_evaluation_metric_value = mean_absolute_percentage_error(y_true,y_pred)
                    best_pipeline = pipe
            else:
                print('Please put either ''R2'' or ''MAE'' or ''RMSE'' or \
                      ''MAPE'' in the evaluation_metric argument')
                
        print('The best scaler and classifier is {} with {} of {}'.
              format(best_pipeline.steps,evaluation_metric,
                     best_evaluation_metric_value))
        
        with open(BEST_PIPELINE_PATH,'wb') as file:
            pickle.dump(best_pipeline,file)
        
        return best_pipeline    

    def dl_simple_model(self,X_train,y_train,cat_or_con,activation_dense='relu',
                        dense_node=128,dropout_rate=0.3):
        
        if cat_or_con=='cat':
            activation_output='softmax'
        elif cat_or_con=='con':
            activation_output='relu'
        else:
            print('Please put either ''cat'' or ''con'' for the cat_or_con argument')
        
        model = Sequential()
        model.add(Input(shape=np.shape(X_train)[1:]))
        model.add(Dense(dense_node,activation=activation_dense))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(dense_node,activation=activation_dense))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(dense_node,activation=activation_dense))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(len(np.unique(y_train,axis=0)),
                            activation=activation_output))
        model.summary()
        
        plot_model(model,show_layer_names=(True),show_shapes=True,
                   to_file=PLOT_PATH)
        
        return model
    
    def dl_lstm_model(self,X_train,y_train,cat_or_con,activation_dense='relu',
                        dense_node=128,dropout_rate=0.3):
        if cat_or_con=='cat':
            activation_output='softmax'
        elif cat_or_con=='con':
            activation_output='relu'
        else:
            print('Please put either ''cat'' or ''con'' for the cat_or_con argument')
        
        
        model = Sequential()   
        model.add(Input(shape=np.shape(X_train)[1:]))
        model.add(Masking())
        model.add(SimpleRNN(2*dense_node,return_sequences=(True)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(2*dense_node,return_sequences=(True)))
        model.add(Dropout(dropout_rate))
        model.add(GRU(2*dense_node))
        model.add(Dense(dense_node,activation=activation_dense))
        model.add(Dropout(dropout_rate))
        model.add(Dense(len(np.unique(y_train,axis=0)),
                             activation=activation_output))
        model.summary()
        
        plot_model(model,show_layer_names=(True),show_shapes=True,
                   to_file=PLOT_PATH)
        
        return model
    
    def dl_model_compilation(self,model,cat_or_con):
        
        if cat_or_con=='cat':
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics='acc')
        elif cat_or_con=='con':
            model.compile(optimizer='adam',
                          loss='mse',
                          metrics='mse')
        else:
            print('Please enter either ''cat'' or ''con'' in the second argument')

    def dl_model_training(self,X_train,X_test,y_train,y_test,model,epochs=10,
                       monitor='val_loss',use_early_callback=False,
                       use_model_checkpoint=False):
        
        tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
        callbacks = [tensorboard_callback]
        
        if use_early_callback==True:
            if epochs <= 30:
                early_callback = EarlyStopping(monitor=monitor,patience=3)
                callbacks.extend([early_callback])
            else:
                early_callback = EarlyStopping(monitor=monitor,
                                               patience=np.floor(0.1*epochs))
                callbacks.extend([early_callback])
        elif use_early_callback==False:
            early_callback=None
        else:
            print('Please put only True or False for use_early_callback argument')
        
        if monitor=='val_acc':
            mode='max'
        elif monitor=='val_loss':
            mode='min'
        else:
            mode='auto'
        
        if use_model_checkpoint==True:
            model_checkpoint = ModelCheckpoint(BEST_MODEL_PATH, monitor=monitor,
                                               save_best_only=(True),
                                               mode=mode,verbose=1)
            callbacks.extend([model_checkpoint])
        elif use_model_checkpoint==False:
            model_checkpoint=None
        else:
            print('Please put only True or False for use_model_checkpoint argument')
        
        hist = model.fit(X_train,y_train,epochs=epochs,verbose=1,
                         validation_data=(X_test,y_test),
                         callbacks=callbacks)
        
        model.save(MODEL_PATH)
        
        return hist
        
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
    
    def dl_plot_hist(self,hist):
        
        keys = list(hist.history.keys())
        
        plt.figure()
        plt.plot(hist.history[keys[0]])
        plt.plot(hist.history[keys[2]])
        plt.xlabel('Epoch')
        plt.legend(['Training '+keys[0],'Validation '+keys[0]])
        plt.show()

        plt.figure()
        plt.plot(hist.history[keys[1]])
        plt.plot(hist.history[keys[3]])
        plt.xlabel('Epoch')
        plt.legend(['Training '+keys[1],'Validation '+keys[1]])
        plt.show()
        
    def classification_report(self,X_test,y_test,best_model,ml_or_dl,
                              use_model_checkpoint=False):
        
        if use_model_checkpoint==True:
            best_model=load_model(BEST_MODEL_PATH)
        elif use_model_checkpoint==False:
            best_model=best_model
        else:
            print('Please put True or False for use_model_checkpoint argument')
            
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
    