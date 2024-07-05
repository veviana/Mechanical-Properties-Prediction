#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import sklearn
from datetime import datetime
import datetime as dt
import joblib

from flask import Flask, request, render_template

from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import metrics

from sklearn.model_selection import train_test_split
import sklearn.model_selection
from sklearn.svm import SVR
from sklearn import svm
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_decomposition import PLSRegression


# # 49

# In[4]:


data_49 = pandas.read_csv('Explanatory_variables//49.csv')


# In[5]:


data_49['DATE'] = pd.to_datetime(data_49['DATE'],format= "%d/%m/%Y")


# In[6]:


data_49 = data_49[(data_49.DATE != '2021-11-16')]


# In[7]:


data_49 = data_49.drop(columns=['DRYER_B1_1'])


# In[8]:


data_49_grouped = data_49.groupby(['DATE']).mean()


# In[9]:


print(data_49_grouped.shape)


# # 48

# In[10]:


data_48 = pandas.read_csv('Explanatory_variables//48.csv')


# In[11]:


data_48['DATE'] = pd.to_datetime(data_48['DATE'],format= "%d/%m/%Y")


# In[12]:


data_48 = data_48.drop(columns=['DRYER_B1_1'])


# In[13]:


data_48_grouped = data_48.groupby(['DATE']).mean()


# # 47

# In[14]:


data_47 = pandas.read_csv('Explanatory_variables//47.csv')


# In[15]:


data_47['DATE'] = pd.to_datetime(data_47['DATE'],format= "%d/%m/%Y")


# In[16]:


data_47 = data_47[(data_47.DATE != '2021-10-18')]


# In[17]:


data_47 = data_47.drop(columns=['DRYER_B1_1'])


# In[18]:


data_47_grouped = data_47.groupby(['DATE']).mean()


# In[19]:


data_47_grouped.head()


# In[20]:


print(data_47_grouped.shape)


# # 46

# In[21]:


data_46 = pandas.read_csv('Explanatory_variables//46.csv')


# In[22]:


data_46['DATE'] = pd.to_datetime(data_46['DATE'],format= "%d/%m/%Y")


# In[23]:


data_46 = data_46[(data_46.DATE != '2021-10-7')]


# In[24]:


data_46 = data_46.drop(columns=['DRYER_B1_1'])


# In[25]:


data_46_grouped = data_46.groupby(['DATE']).mean()


# In[26]:


print(data_46_grouped.shape)


# # 45

# In[27]:


data_45 = pandas.read_csv('Explanatory_variables//45.csv')


# In[28]:


data_45['DATE'] = pd.to_datetime(data_45['DATE'],format= "%d/%m/%Y")


# In[29]:


data_45 = data_45.drop(columns=['DRYER_B1_1'])


# In[30]:


data_45_grouped = data_45.groupby(['DATE']).mean()


# In[31]:


print(data_45_grouped.shape)


# # 44 

# In[32]:


data_44 = pandas.read_csv('Explanatory_variables//44.csv')


# In[33]:


data_44['DATE'] = pd.to_datetime(data_44['DATE'],format= "%d/%m/%Y")


# In[34]:


data_44 = data_44[(data_44.DATE != '2021-08-26')]


# In[35]:


data_44 = data_44.drop(columns=['DRYER_B1_1'])


# In[36]:


data_44_grouped = data_44.groupby(['DATE']).mean()


# In[37]:


print(data_44_grouped.shape)


# # 43

# In[38]:


data_43 = pandas.read_csv('Explanatory_variables//43.csv')


# In[39]:


data_43['DATE'] = pd.to_datetime(data_43['DATE'],format= "%d/%m/%Y")


# In[40]:


data_43 = data_43[(data_43.DATE != '2021-08-16')]


# In[41]:


data_43 = data_43.drop(columns=['DRYER_B1_1'])


# In[42]:


data_43_grouped = data_43.groupby(['DATE']).mean()


# In[43]:


print(data_43_grouped.shape)


# # 42

# In[44]:


data_42 = pandas.read_csv('Explanatory_variables//42.csv')


# In[45]:


data_42['DATE'] = pd.to_datetime(data_42['DATE'],format= "%d/%m/%Y")


# In[46]:


data_42 = data_42.drop(columns=['DRYER_B1_1'])


# In[47]:


data_42_grouped = data_42.groupby(['DATE']).mean()


# In[48]:


print(data_42_grouped.shape)


# # 41

# In[49]:


data_41 = pandas.read_csv('Explanatory_variables//41.csv')


# In[50]:


data_41['DATE'] = pd.to_datetime(data_41['DATE'],format= "%d/%m/%Y")


# In[51]:


data_41 = data_41[(data_41.DATE != '2021-08-06')]


# In[52]:


data_41 = data_41.drop(columns=['DRYER_B1_1'])


# In[53]:


data_41_grouped = data_41.groupby(['DATE']).mean()


# In[54]:


print(data_41_grouped.shape)


# # 40

# In[55]:


data_40 = pandas.read_csv('Explanatory_variables//40.csv')


# In[56]:


data_40['DATE'] = pd.to_datetime(data_40['DATE'],format= "%d/%m/%Y")


# In[57]:


data_40 = data_40.drop(columns=['DRYER_B1_1'])


# In[58]:


data_40_grouped = data_40.groupby(['DATE']).mean()


# In[59]:


print(data_40_grouped.shape)


# # 36

# In[60]:


data_36 = pandas.read_csv('Explanatory_variables//36.csv')


# In[61]:


data_36['DATE'] = pd.to_datetime(data_36['DATE'],format= "%d/%m/%Y")


# In[62]:


data_36 = data_36.drop(columns=['DRYER_B1_1'])


# In[63]:


data_36_grouped = data_36.groupby(['DATE']).mean()


# In[64]:


print(data_36_grouped.shape)


# # 35

# In[65]:


data_35 = pandas.read_csv('Explanatory_variables//35.csv')


# In[66]:


data_35['DATE'] = pd.to_datetime(data_35['DATE'],format= "%d/%m/%Y")


# In[67]:


data_35 = data_35.drop(columns=['DRYER_B1_1'])


# In[68]:


data_35_grouped = data_35.groupby(['DATE']).mean()


# In[69]:


print(data_35_grouped.shape)


# # 34

# In[70]:


data_34 = pandas.read_csv('Explanatory_variables//34.csv')


# In[71]:


data_34['DATE'] = pd.to_datetime(data_34['DATE'],format= "%d/%m/%Y")


# In[72]:


data_34 = data_34[(data_34.DATE != '2021-04-24')]


# In[73]:


data_34 = data_34.drop(columns=['DRYER_B1_1'])


# In[74]:


data_34_grouped = data_34.groupby(['DATE']).mean()


# In[75]:


print(data_34_grouped.shape)


# # 33

# In[76]:


data_33 = pandas.read_csv('Explanatory_variables//33.csv')


# In[77]:


data_33['DATE'] = pd.to_datetime(data_33['DATE'],format= "%d/%m/%Y")


# In[78]:


data_33 = data_33.drop(columns=['DRYER_B1_1'])


# In[79]:


data_33_grouped = data_33.groupby(['DATE']).mean()


# In[80]:


print(data_33_grouped.shape)


# # 32

# In[81]:


data_32 = pandas.read_csv('Explanatory_variables//32.csv')


# In[82]:


data_32['DATE'] = pd.to_datetime(data_32['DATE'],format= "%d/%m/%Y")


# In[83]:


data_32 = data_32.drop(columns=['DRYER_B1_1'])


# In[84]:


data_32_grouped = data_32.groupby(['DATE']).mean()


# In[85]:


print(data_32_grouped.shape)


# # Group together

# In[86]:


df = pd.concat([data_32_grouped, data_33_grouped,data_34_grouped,data_35_grouped,
                data_36_grouped,data_40_grouped,data_41_grouped,data_42_grouped,data_43_grouped,
               data_44_grouped,data_45_grouped,data_46_grouped,data_47_grouped,data_48_grouped,
               data_49_grouped],join='outer')


# In[87]:


print(df.shape)


# # Start

# # Y1

# In[88]:


data_Y1 = pandas.read_csv('Objective_variable//Y1.csv')
data_Y1['Production date'] = pd.to_datetime(data_Y1['Production date'], format='%Y%m%d')
filtered_dataY1 = data_Y1[(data_Y1.GROUP >= 32)]


# In[89]:


#Assign data to x and y
y = filtered_dataY1['Start']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y1, y_test_y1 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[90]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y1.shape)
print(y_test_y1.shape)


# # ML Algorithms (Y1)

# In[91]:


LR = LinearRegression()
LR.fit(x_train, y_train_y1)
y_pred_y1 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
LRRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[92]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y1)
y_pred_y1 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
SVRRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[93]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y1)
y_pred_y1 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
RFRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[94]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y1)
y_pred_y1 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
GBRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[95]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y1)
y_pred_y1 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
ABRRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[96]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y1)
y_pred_y1 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
PLSRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[97]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[98]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y1) 

# In[82]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "RandomForest"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)


    regressor_obj.fit(x_train, y_train_y1)
    y_pred_y1 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y1, y_pred_y1)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[83]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["SVR", "PLSR"])
    if classifier_name == "SVR":
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c, gamma="auto")
    else:
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)


    regressor_obj.fit(x_train, y_train_y1)
    y_pred_y1 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y1, y_pred_y1)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[84]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["LinearRegression", "GBR"])
    if classifier_name == "LinearRegression":
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
    else:
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)


    regressor_obj.fit(x_train, y_train_y1)
    y_pred_y1 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y1, y_pred_y1)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model & saving trained model (Random Forest) - Y1

# In[102]:


rf = RandomForestRegressor(n_estimators = 10, max_depth = 10)

# Train the model on training data
rf.fit(x_train, y_train_y1)
y_pred_y1 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
RFRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', RFRSV)
print('Mean Squared Error:', RFMSE)

# Save the trained model to a file so we can use it in other programs
joblib.dump(rf, 'models//RFY1start.pkl')


# # Predicted - Observed Plot (Y1)

# In[86]:


rf = RandomForestRegressor(n_estimators = 10, max_depth = 10)

# Train the model on training data
rf.fit(x_train, y_train_y1)
y_pred_y1 = rf.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y1, y_pred_y1)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[87]:


rf = RandomForestRegressor(n_estimators = 10, max_depth = 10)
rf.fit(x_train, y_train_y1)
y_pred_y1 = rf.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y1, y_pred_y1)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# # Y2

# In[97]:


data_Y2 = pandas.read_csv('Objective_variable//Y2.csv')
data_Y2['Production date'] = pd.to_datetime(data_Y2['Production date'], format='%Y%m%d')
filtered_dataY2 = data_Y2[(data_Y2.GROUP >= 32)]


# In[98]:


#Assign data to x and y
y = filtered_dataY2['Start']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y2, y_test_y2 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[99]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y2.shape)
print(y_test_y2.shape)


# # ML Algorithms (Y2)

# In[100]:


LR = LinearRegression()
LR.fit(x_train, y_train_y2)
y_pred_y2 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
LRRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[101]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y2)
y_pred_y2 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
SVRRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[102]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y2)
y_pred_y2 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
RFRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[103]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y2)
y_pred_y2 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
GBRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[104]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y2)
y_pred_y2 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
ABRRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[105]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y2)
y_pred_y2 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
PLSRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[106]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[107]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y2) 

# In[99]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["PLSR", "SVR"])
    if classifier_name == "PLSR":
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)
    else:
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c, gamma="auto")


    regressor_obj.fit(x_train, y_train_y2)
    y_pred_y2 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y2, y_pred_y2)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[100]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "RandomForest"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)


    regressor_obj.fit(x_train, y_train_y2)
    y_pred_y2 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y2, y_pred_y2)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[101]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["LinearRegression", "PLSR"])
    if classifier_name == "LinearRegression":
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
    else:
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)

    regressor_obj.fit(x_train, y_train_y2)
    y_pred_y2 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y2, y_pred_y2)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model (Partial Least-Square) - Y2

# In[102]:


PLS = PLSRegression(n_components=1,max_iter=269)
PLS.fit(x_train, y_train_y2)
y_pred_y2 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
PLSRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', PLSRSV)
print('Mean Squared Error:', PLSMSE)


# # Predicted - Observed Plot (Y2)

# In[103]:


PLS = PLSRegression(n_components=1,max_iter=269)
PLS.fit(x_train, y_train_y2)
y_pred_y2 = PLS.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y2, y_pred_y2)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[180]:


PLS = PLSRegression(n_components=1,max_iter=269)
PLS.fit(x_train, y_train_y2)
y_pred_y2 = PLS.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y2, y_pred_y2)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# # Y3

# In[105]:


data_Y3 = pandas.read_csv('Objective_variable//Y3.csv')
data_Y3['Production date'] = pd.to_datetime(data_Y3['Production date'], format='%Y%m%d')
filtered_dataY3 = data_Y3[(data_Y3.GROUP >= 32)]


# In[106]:


#Assign data to x and y
y = filtered_dataY3['Start']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y3, y_test_y3 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[107]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y3.shape)
print(y_test_y3.shape)


# # ML Algorithms (Y3)

# In[108]:


LR = LinearRegression()
LR.fit(x_train, y_train_y3)
y_pred_y3 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
LRRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[109]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y3)
y_pred_y3 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
SVRRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[110]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y3)
y_pred_y3 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
RFRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[111]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y3)
y_pred_y3 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
GBRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[112]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y3)
y_pred_y3 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
ABRRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[113]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y3)
y_pred_y3 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
PLSRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[114]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[115]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y3) 

# In[116]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["SVR", "PLSR"])
    if classifier_name == "SVR":
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c, gamma="auto")
    else:
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)


    regressor_obj.fit(x_train, y_train_y3)
    y_pred_y3 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y3, y_pred_y3)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[117]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "RandomForest"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)


    regressor_obj.fit(x_train, y_train_y3)
    y_pred_y3 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y3, y_pred_y3)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[118]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["LinearRegression", "GBR"])
    if classifier_name == "LinearRegression":
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
    else:
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)


    regressor_obj.fit(x_train, y_train_y3)
    y_pred_y3 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y3, y_pred_y3)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model (AdaBoost) - Y3

# In[121]:


ABR = AdaBoostRegressor(n_estimators = 1904)
ABR.fit(x_train, y_train_y3)
y_pred_y3 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
ABRRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', ABRRSV)
print('Mean Squared Error:', ABRMSE)


# # Predicted - Observed Plot (Y3)

# In[122]:


ABR = AdaBoostRegressor(n_estimators = 1904)
ABR.fit(x_train, y_train_y3)
y_pred_y3 = ABR.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y3, y_pred_y3)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[179]:


ABR = AdaBoostRegressor(n_estimators = 1904)
ABR.fit(x_train, y_train_y3)
y_pred_y3 = ABR.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y3, y_pred_y3)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# # Y4

# In[124]:


data_Y4 = pandas.read_csv('Objective_variable//Y4.csv')
data_Y4['Production date'] = pd.to_datetime(data_Y4['Production date'], format='%Y%m%d')
filtered_dataY4 = data_Y4[(data_Y4.GROUP >= 32)]


# In[125]:


#Assign data to x and y
y = filtered_dataY4['Start']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y4, y_test_y4 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[126]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y4.shape)
print(y_test_y4.shape)


# # ML Algorithms (Y4)

# In[127]:


LR = LinearRegression()
LR.fit(x_train, y_train_y4)
y_pred_y4 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
LRRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[128]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y4)
y_pred_y4 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
SVRRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[129]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y4)
y_pred_y4 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
RFRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[130]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y4)
y_pred_y4 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
GBRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[131]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y4)
y_pred_y4 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
ABRRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[132]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y4)
y_pred_y4 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
PLSRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[133]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[134]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y4) 

# In[135]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "RandomForest"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)


    regressor_obj.fit(x_train, y_train_y4)
    y_pred_y4 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y4, y_pred_y4)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[136]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["SVR", "GBR"])
    if classifier_name == "SVR":
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c, gamma="auto")
    else:
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)


    regressor_obj.fit(x_train, y_train_y4)
    y_pred_y4 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y4, y_pred_y4)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[137]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["PLSR", "LinearRegression"])
    if classifier_name == "PLSR":
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)

    else:
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
        

    regressor_obj.fit(x_train, y_train_y4)
    y_pred_y4 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y4, y_pred_y4)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model (AdaBoost) - Y4

# In[138]:


ABR = AdaBoostRegressor(n_estimators = 1282)
ABR.fit(x_train, y_train_y4)
y_pred_y4 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
ABRRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', ABRRSV)
print('Mean Squared Error:', ABRMSE)


# # Predicted - Observed Plot (Y4)

# In[139]:


ABR = AdaBoostRegressor(n_estimators = 1282)
ABR.fit(x_train, y_train_y4)
y_pred_y4 = ABR.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y4, y_pred_y4)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[178]:


ABR = AdaBoostRegressor(n_estimators = 1282)
ABR.fit(x_train, y_train_y4)
y_pred_y4 = ABR.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y4, y_pred_y4)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# # Middle

# In[141]:


data_Y1 = pandas.read_csv('Objective_variable//Y1.csv')
data_Y1['Production date'] = pd.to_datetime(data_Y1['Production date'], format='%Y%m%d')
filtered_dataY1 = data_Y1[(data_Y1.GROUP >= 32)]


# In[142]:


#Assign data to x and y
y = filtered_dataY1['Middle']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y1, y_test_y1 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[143]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y1.shape)
print(y_test_y1.shape)


# # Y1

# In[144]:


LR = LinearRegression()
LR.fit(x_train, y_train_y1)
y_pred_y1 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
LRRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[145]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y1)
y_pred_y1 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
SVRRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[146]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y1)
y_pred_y1 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
RFRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[147]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y1)
y_pred_y1 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
GBRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[148]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y1)
y_pred_y1 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
ABRRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[149]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y1)
y_pred_y1 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
PLSRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[150]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[151]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y1) 

# In[152]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "GBR"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)


    regressor_obj.fit(x_train, y_train_y1)
    y_pred_y1 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y1, y_pred_y1)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[153]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["RandomForest", "PLSR"])
    if classifier_name == "RandomForest":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)
    else:
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)


    regressor_obj.fit(x_train, y_train_y1)
    y_pred_y1 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y1, y_pred_y1)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[154]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["LinearRegression", "SVR"])
    if classifier_name == "LinearRegression":
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
    else:
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c, gamma="auto")

    regressor_obj.fit(x_train, y_train_y1)
    y_pred_y1 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y1, y_pred_y1)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model (Gradient Boosting) - Y1

# In[157]:


GBR = GradientBoostingRegressor(n_estimators = 2875, max_depth = 4)
GBR.fit(x_train, y_train_y1)
y_pred_y1 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
GBRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', GBRSV)
print('Mean Squared Error:', GBRMSE)


# # Predicted - Observed Plot (Y1)

# In[158]:


GBR = GradientBoostingRegressor(n_estimators = 2875, max_depth = 4)
GBR.fit(x_train, y_train_y1)
y_pred_y1 = GBR.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y1, y_pred_y1)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[159]:


GBR = GradientBoostingRegressor(n_estimators = 2875, max_depth = 4)
GBR.fit(x_train, y_train_y1)
y_pred_y1 = GBR.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y1, y_pred_y1)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# # Y2

# In[160]:


data_Y2 = pandas.read_csv('Objective_variable//Y2.csv')
data_Y2['Production date'] = pd.to_datetime(data_Y2['Production date'], format='%Y%m%d')
filtered_dataY2 = data_Y2[(data_Y2.GROUP >= 32)]


# In[161]:


#Assign data to x and y
y = filtered_dataY2['Middle']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y2, y_test_y2 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[162]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y2.shape)
print(y_test_y2.shape)


# In[163]:


LR = LinearRegression()
LR.fit(x_train, y_train_y2)
y_pred_y2 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
LRRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[164]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y2)
y_pred_y2 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
SVRRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[165]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y2)
y_pred_y2 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
RFRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[166]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y2)
y_pred_y2 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
GBRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[167]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y2)
y_pred_y2 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
ABRRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[168]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y2)
y_pred_y2 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
PLSRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[169]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[170]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y2) 

# In[171]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "RandomForest"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)


    regressor_obj.fit(x_train, y_train_y2)
    y_pred_y2 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y2, y_pred_y2)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[172]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["PLSR", "SVR"])
    if classifier_name == "PLSR":
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)
    else:
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c, gamma="auto")


    regressor_obj.fit(x_train, y_train_y2)
    y_pred_y2 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y2, y_pred_y2)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[173]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["LinearRegression", "GBR"])
    if classifier_name == "LinearRegression":
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
    else:
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)

    regressor_obj.fit(x_train, y_train_y2)
    y_pred_y2 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y2, y_pred_y2)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model (AdaBoost) - Y2

# In[174]:


ABR = AdaBoostRegressor(n_estimators = 186)
ABR.fit(x_train, y_train_y2)
y_pred_y2 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
ABRRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', ABRRSV)
print('Mean Squared Error:', ABRMSE)


# # Predicted - Observed Plot (Y2)

# In[175]:


ABR = AdaBoostRegressor(n_estimators = 186)
ABR.fit(x_train, y_train_y2)
y_pred_y2 = ABR.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y2, y_pred_y2)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[177]:


ABR = AdaBoostRegressor(n_estimators = 186)
ABR.fit(x_train, y_train_y2)
y_pred_y2 = ABR.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y2, y_pred_y2)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# # Y3

# In[181]:


data_Y3 = pandas.read_csv('Objective_variable//Y3.csv')
data_Y3['Production date'] = pd.to_datetime(data_Y3['Production date'], format='%Y%m%d')
filtered_dataY3 = data_Y3[(data_Y3.GROUP >= 32)]


# In[182]:


#Assign data to x and y
y = filtered_dataY3['Middle']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y3, y_test_y3 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[183]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y3.shape)
print(y_test_y3.shape)


# In[184]:


LR = LinearRegression()
LR.fit(x_train, y_train_y3)
y_pred_y3 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
LRRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[185]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y3)
y_pred_y3 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
SVRRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[186]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y3)
y_pred_y3 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
RFRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[187]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y3)
y_pred_y3 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
GBRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[188]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y3)
y_pred_y3 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
ABRRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[189]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y3)
y_pred_y3 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
PLSRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[190]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[191]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y3) 

# In[192]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["PLSR", "SVR"])
    if classifier_name == "PLSR":
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)
    else:
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c)


    regressor_obj.fit(x_train, y_train_y3)
    y_pred_y3 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y3, y_pred_y3)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[193]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "RandomForest"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)


    regressor_obj.fit(x_train, y_train_y3)
    y_pred_y3 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y3, y_pred_y3)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[194]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["LinearRegression", "GBR"])
    if classifier_name == "LinearRegression":
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
    else:
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)


    regressor_obj.fit(x_train, y_train_y3)
    y_pred_y3 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y3, y_pred_y3)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model (Partial Least-Square) - Y3

# In[195]:


PLS = PLSRegression(n_components = 6, max_iter = 654)
PLS.fit(x_train, y_train_y3)
y_pred_y3 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
PLSRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', PLSRSV)
print('Mean Squared Error:', PLSMSE)


# # Predicted - Observed Plot (Y3)

# In[196]:


PLS = PLSRegression(n_components = 6, max_iter = 654)
PLS.fit(x_train, y_train_y3)
y_pred_y3 = PLS.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y3, y_pred_y3)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[197]:


PLS = PLSRegression(n_components = 6, max_iter = 654)
PLS.fit(x_train, y_train_y3)
y_pred_y3 = PLS.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y3, y_pred_y3)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# # Y4

# In[198]:


data_Y4 = pandas.read_csv('Objective_variable//Y4.csv')
data_Y4['Production date'] = pd.to_datetime(data_Y4['Production date'], format='%Y%m%d')
filtered_dataY4 = data_Y4[(data_Y4.GROUP >= 32)]


# In[199]:


#Assign data to x and y
y = filtered_dataY4['Middle']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y4, y_test_y4 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[200]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y4.shape)
print(y_test_y4.shape)


# In[201]:


LR = LinearRegression()
LR.fit(x_train, y_train_y4)
y_pred_y4 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
LRRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[202]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y4)
y_pred_y4 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
SVRRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[203]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y4)
y_pred_y4 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
RFRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[204]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y4)
y_pred_y4 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
GBRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[205]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y4)
y_pred_y4 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
ABRRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[206]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y4)
y_pred_y4 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
PLSRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[207]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[208]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y4) 

# In[209]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["PLSR", "SVR"])
    if classifier_name == "PLSR":
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)

    else:
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c, gamma="auto")
        

    regressor_obj.fit(x_train, y_train_y4)
    y_pred_y4 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y4, y_pred_y4)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[210]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "RandomForest"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)


    regressor_obj.fit(x_train, y_train_y4)
    y_pred_y4 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y4, y_pred_y4)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[211]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["GBR", "LinearRegression"])
    if classifier_name == "GBR":
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)

    else:
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
        

    regressor_obj.fit(x_train, y_train_y4)
    y_pred_y4 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y4, y_pred_y4)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model (Partial Least-Square) - Y4

# In[214]:


PLS = PLSRegression(n_components = 11, max_iter = 2113)
PLS.fit(x_train, y_train_y4)
y_pred_y4 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
PLSRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', PLSRSV)
print('Mean Squared Error:', PLSMSE)


# # Predicted - Observed Plot (Y4)

# In[215]:


PLS = PLSRegression(n_components = 11, max_iter = 2113)
PLS.fit(x_train, y_train_y4)
y_pred_y4 = PLS.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y4, y_pred_y4)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[217]:


PLS = PLSRegression(n_components = 11, max_iter = 2113)
PLS.fit(x_train, y_train_y4)
y_pred_y4 = PLS.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y4, y_pred_y4)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# # End

# # Y1

# In[218]:


data_Y1 = pandas.read_csv('Objective_variable//Y1.csv')
data_Y1['Production date'] = pd.to_datetime(data_Y1['Production date'], format='%Y%m%d')
filtered_dataY1 = data_Y1[(data_Y1.GROUP >= 32)]


# In[219]:


#Assign data to x and y
y = filtered_dataY1['End']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y1, y_test_y1 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[220]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y1.shape)
print(y_test_y1.shape)


# In[221]:


LR = LinearRegression()
LR.fit(x_train, y_train_y1)
y_pred_y1 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
LRRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[222]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y1)
y_pred_y1 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
SVRRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[223]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y1)
y_pred_y1 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
RFRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[224]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y1)
y_pred_y1 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
GBRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[225]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y1)
y_pred_y1 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
ABRRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[226]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y1)
y_pred_y1 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
PLSRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y1, y_pred_y1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y1, y_pred_y1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y1, y_pred_y1)))


# In[227]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[228]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y1) 

# In[229]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "RandomForest"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)

    regressor_obj.fit(x_train, y_train_y1)
    y_pred_y1 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y1, y_pred_y1)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[230]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["GBR", "SVR"])
    if classifier_name == "GBR":
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)

    else:
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c, gamma="auto")

    regressor_obj.fit(x_train, y_train_y1)
    y_pred_y1 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y1, y_pred_y1)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[231]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["LinearRegression", "PLSR"])
    if classifier_name == "LinearRegression":
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
    else:
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)


    regressor_obj.fit(x_train, y_train_y1)
    y_pred_y1 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y1, y_pred_y1)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model (AdaBoost) - Y1

# In[232]:


ABR = AdaBoostRegressor(n_estimators = 177)
ABR.fit(x_train, y_train_y1)
y_pred_y1 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y1, y_pred_y1)
ABRRSV = r2_score(y_test_y1, y_pred_y1)

print('R Squared Value:', ABRRSV)
print('Mean Squared Error:', ABRMSE)


# # Predicted - Observed Plot (Y1)

# In[233]:


ABR = AdaBoostRegressor(n_estimators = 177)
ABR.fit(x_train, y_train_y1)
y_pred_y1 = ABR.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y1, y_pred_y1)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[234]:


ABR = AdaBoostRegressor(n_estimators = 177)
ABR.fit(x_train, y_train_y1)
y_pred_y1 = ABR.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y1, y_pred_y1)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# # Y2

# In[77]:


data_Y2 = pandas.read_csv('Objective_variable//Y2.csv')
data_Y2['Production date'] = pd.to_datetime(data_Y2['Production date'], format='%Y%m%d')
filtered_dataY2 = data_Y2[(data_Y2.GROUP >= 32)]


# In[78]:


#Assign data to x and y
y = filtered_dataY2['End']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y2, y_test_y2 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[79]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y2.shape)
print(y_test_y2.shape)


# In[80]:


LR = LinearRegression()
LR.fit(x_train, y_train_y2)
y_pred_y2 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
LRRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[81]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y2)
y_pred_y2 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
SVRRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[82]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y2)
y_pred_y2 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
RFRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[83]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y2)
y_pred_y2 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
GBRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[84]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y2)
y_pred_y2 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
ABRRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[85]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y2)
y_pred_y2 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
PLSRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y2, y_pred_y2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y2, y_pred_y2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y2, y_pred_y2)))


# In[86]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[87]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y2) 

# In[88]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "SVR"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c, gamma="auto")



    regressor_obj.fit(x_train, y_train_y2)
    y_pred_y2 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y2, y_pred_y2)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[89]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["PLSR", "RandomForest"])
    if classifier_name == "PLSR":
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)



    regressor_obj.fit(x_train, y_train_y2)
    y_pred_y2 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y2, y_pred_y2)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[90]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["LinearRegression", "GBR"])
    if classifier_name == "LinearRegression":
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
    else:
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)

    regressor_obj.fit(x_train, y_train_y2)
    y_pred_y2 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y2, y_pred_y2)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model (AdaBoost) - Y2

# In[91]:


ABR = AdaBoostRegressor(n_estimators = 534)
ABR.fit(x_train, y_train_y2)
y_pred_y2 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y2, y_pred_y2)
ABRRSV = r2_score(y_test_y2, y_pred_y2)

print('R Squared Value:', ABRRSV)
print('Mean Squared Error:', ABRMSE)


# # Predicted - Observed Plot (Y2)

# In[92]:


ABR = AdaBoostRegressor(n_estimators = 534)
ABR.fit(x_train, y_train_y2)
y_pred_y2 = ABR.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y2, y_pred_y2)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[93]:


ABR = AdaBoostRegressor(n_estimators = 534)
ABR.fit(x_train, y_train_y2)
y_pred_y2 = ABR.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y2, y_pred_y2)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# # Y3

# In[94]:


data_Y3 = pandas.read_csv('Objective_variable//Y3.csv')
data_Y3['Production date'] = pd.to_datetime(data_Y3['Production date'], format='%Y%m%d')
filtered_dataY3 = data_Y3[(data_Y3.GROUP >= 32)]


# In[95]:


#Assign data to x and y
y = filtered_dataY3['End']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y3, y_test_y3 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[96]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y3.shape)
print(y_test_y3.shape)


# In[97]:


LR = LinearRegression()
LR.fit(x_train, y_train_y3)
y_pred_y3 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
LRRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[98]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y3)
y_pred_y3 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
SVRRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[99]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y3)
y_pred_y3 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
RFRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[100]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y3)
y_pred_y3 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
GBRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[101]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y3)
y_pred_y3 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
ABRRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[102]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y3)
y_pred_y3 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
PLSRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y3, y_pred_y3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y3, y_pred_y3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y3, y_pred_y3)))


# In[103]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[104]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y3) 

# In[105]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["RandomForest", "PLSR"])
    if classifier_name == "RandomForest":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)
    else:
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)


    regressor_obj.fit(x_train, y_train_y3)
    y_pred_y3 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y3, y_pred_y3)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[106]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "SVR"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c, gamma="auto")



    regressor_obj.fit(x_train, y_train_y3)
    y_pred_y3 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y3, y_pred_y3)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[108]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["LinearRegression", "GBR"])
    if classifier_name == "LinearRegression":
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
    else:
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)

    regressor_obj.fit(x_train, y_train_y3)
    y_pred_y3 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y3, y_pred_y3)
    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model (Partial Least-Square) - Y3

# In[109]:


PLS = PLSRegression(n_components = 5, max_iter = 2459)
PLS.fit(x_train, y_train_y3)
y_pred_y3 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y3, y_pred_y3)
PLSRSV = r2_score(y_test_y3, y_pred_y3)

print('R Squared Value:', PLSRSV)
print('Mean Squared Error:', PLSMSE)


# # Predicted - Observed Plot (Y3)

# In[110]:


PLS = PLSRegression(n_components = 5, max_iter = 2459)
PLS.fit(x_train, y_train_y3)
y_pred_y3 = PLS.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y3, y_pred_y3)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[111]:


PLS = PLSRegression(n_components = 5, max_iter = 2459)
PLS.fit(x_train, y_train_y3)
y_pred_y3 = PLS.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y3, y_pred_y3)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# # Y4

# In[112]:


data_Y4 = pandas.read_csv('Objective_variable//Y4.csv')
data_Y4['Production date'] = pd.to_datetime(data_Y4['Production date'], format='%Y%m%d')
filtered_dataY4 = data_Y4[(data_Y4.GROUP >= 32)]


# In[113]:


#Assign data to x and y
y = filtered_dataY4['End']

cols = [col for col in df.columns]
x = df[cols]

x_train, x_test = train_test_split(x, test_size = 0.3, random_state = 7)
y_train_y4, y_test_y4 = train_test_split(y, test_size = 0.3, random_state = 7)


# In[114]:


print(x_train.shape)
print(x_test.shape)
print(y_train_y4.shape)
print(y_test_y4.shape)


# In[115]:


LR = LinearRegression()
LR.fit(x_train, y_train_y4)
y_pred_y4 = LR.predict(x_test)

#Lower MSE = better model. Higher R^2 = better model
LRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
LRRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', LRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[116]:


SVR = svm.SVR()
SVR.fit(x_train, y_train_y4)
y_pred_y4 = SVR.predict(x_test)

#Lower MSE = better model
SVRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
SVRRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', SVRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[117]:


# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train_y4)
y_pred_y4 = rf.predict(x_test)

#Lower MSE = better model
RFMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
RFRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', RFRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[118]:


GBR = GradientBoostingRegressor(random_state = 42)
GBR.fit(x_train, y_train_y4)
y_pred_y4 = GBR.predict(x_test)

#Lower MSE = better model
GBRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
GBRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', GBRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[119]:


ABR = AdaBoostRegressor(random_state = 42)
ABR.fit(x_train, y_train_y4)
y_pred_y4 = ABR.predict(x_test)

#Lower MSE = better model
ABRMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
ABRRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', ABRRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[120]:


PLS = PLSRegression()
PLS.fit(x_train, y_train_y4)
y_pred_y4 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
PLSRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', PLSRSV)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_y4, y_pred_y4))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_y4, y_pred_y4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_y4, y_pred_y4)))


# In[121]:


#empty array
Model_MSE = []
Model_R2 = []
Model = []

#Append the name of models into the empty array 'Model'
Model.append(('Linear Regression'))
Model.append(('Support Vector Regression'))
Model.append(('Random Forest Regression'))
Model.append(('Gradient Boosting Regressor'))
Model.append(('AdaBoost Regressor'))
Model.append(('Partial Least-Square Regression'))

#Append the model accuracy into the empty array 'Model_Accuracy'
Model_MSE.append((LRMSE))
Model_MSE.append((SVRMSE))
Model_MSE.append((RFMSE))
Model_MSE.append((GBRMSE))
Model_MSE.append((ABRMSE))
Model_MSE.append((PLSMSE))

Model_R2.append((LRRSV))
Model_R2.append((SVRRSV))
Model_R2.append((RFRSV))
Model_R2.append((GBRSV))
Model_R2.append((ABRRSV))
Model_R2.append((PLSRSV))


# In[122]:


d = {'Mean Squared Error':Model_MSE,'R-Squared Value':Model_R2}
da = pd.DataFrame(d, index = Model)
da.sort_values(by=['Mean Squared Error'])


# # Hyperparameter Tuning With Optuna (Y4) 

# In[123]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["PLSR", "SVR"])
    if classifier_name == "PLSR":
        components = trial.suggest_int("components", 1, 20)
        iteration = trial.suggest_int("iteration", 1, 3000)
        regressor_obj = PLSRegression(n_components=components, max_iter=iteration)

    else:
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c, gamma="auto")
        

    regressor_obj.fit(x_train, y_train_y4)
    y_pred_y4 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y4, y_pred_y4)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[124]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["Adaboost", "RandomForest"])
    if classifier_name == "Adaboost":
        estimators = trial.suggest_int("estimators", 1, 3000)
        regressor_obj = AdaBoostRegressor(n_estimators=estimators)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth, n_estimators=10)


    regressor_obj.fit(x_train, y_train_y4)
    y_pred_y4 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y4, y_pred_y4)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# In[125]:


def objective(trial):

    classifier_name = trial.suggest_categorical("regressor", ["GBR", "LinearRegression"])
    if classifier_name == "GBR":
        estimators = trial.suggest_int("estimators", 1, 3000)
        depths = trial.suggest_int("depths", 1, 20)
        regressor_obj = GradientBoostingRegressor(n_estimators=estimators, max_depth=depths)

    else:
        intercepts = trial.suggest_categorical('intercepts', [True, False])
        jobs = trial.suggest_int('jobs', 0, 100)
        regressor_obj = LinearRegression(n_jobs=jobs, fit_intercept=intercepts)
        

    regressor_obj.fit(x_train, y_train_y4)
    y_pred_y4 = regressor_obj.predict(x_test)
    
    error = sklearn.metrics.mean_squared_error(y_test_y4, y_pred_y4)

    return error

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=200)  # Invoke optimization of the objective function.
print(study.best_trial.value)


# # Make Prediction with best model (Partial Least-Square) - Y4

# In[126]:


PLS = PLSRegression(n_components = 6, max_iter = 2086)
PLS.fit(x_train, y_train_y4)
y_pred_y4 = PLS.predict(x_test)

#Lower MSE = better model
PLSMSE = metrics.mean_squared_error(y_test_y4, y_pred_y4)
PLSRSV = r2_score(y_test_y4, y_pred_y4)

print('R Squared Value:', PLSRSV)
print('Mean Squared Error:', PLSMSE)


# # Predicted - Observed Plot (Y4)

# In[127]:


PLS = PLSRegression(n_components = 6, max_iter = 2086)
PLS.fit(x_train, y_train_y4)
y_pred_y4 = PLS.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test_y4, y_pred_y4)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[128]:


PLS = PLSRegression(n_components = 6, max_iter = 2086)
PLS.fit(x_train, y_train_y4)
y_pred_y4 = PLS.predict(x_train).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_train_y4, y_pred_y4)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [2,3]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# In[ ]:





# In[ ]:


# Define the '/predict' route to:
# - Get form data and convert them to float values
# - Convert form data to numpy array
# - Pass form data to model for prediction

@app.route('/predict',methods=['POST'])
def predict():

    columns = ["DRYER_A7_3", "DRYER_A2_1", "DRYER_A2_2", "DRYER_A6_1", "DRYER_A2_3", "DRYER_A1_1", "DRYER_A1_3",
               "DRYER_B1_2", "DRYER_B1_3", "DRYER_B2_1", "DRYER_A4_3", "DRYER_B2_2", "DRYER_B2_3", "DRYER_P1", 
               "DRYER_A7_1", "DRYER_A7_2", "DRYER_A5_2", "DRYER_A5_3", "DRYER_A6_2", "DRYER_A6_3", "DRYER_A4_2", 
               "DRYER_A5_1", "EPC_ROOM", "DRYER_A3_1", "DRYER_A3_2", "DRYER_A3_3", "DRYER_A4_1", "DRYER_A1_2"]


    df = pd.DataFrame.from_dict(zip(columns,request.form.values()))
    correct_df = df.T


    new_header = correct_df.iloc[0] #grab the first row for the header
    correct_df = correct_df[1:] #take the data less the header row
    correct_df.columns = new_header #set the header row as the df header

    for i in columns:
        correct_df[i] = pd.to_numeric(correct_df[i])

    #print(correct_df.dtypes)
    predicted_customer_churn = model.predict(correct_df)


    #predicted_value = predicted_customer_churn[0]

	# Format prediction text for display in "index.html"
    return render_template('Y1StartPrediction.html', Churn_prediction='This customer will churn or not: {}'.format(predicted_customer_churn[0]))


# In[ ]:





# In[ ]:




