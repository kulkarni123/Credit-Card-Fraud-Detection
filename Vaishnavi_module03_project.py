# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:30:20 2022

@author: Vaishu
"""
#Importing all neccessary Libraries
import warnings
warnings.filterwarnings('ignore')
from sklearn.experimental import enable_halving_search_cv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
from sklearn.model_selection import train_test_split, cross_validate, HalvingGridSearchCV,GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold,RandomizedSearchCV,cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score,accuracy_score,fbeta_score,classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

#%%% Reading the data set using Pandas

df=pd.read_csv(r"C:\Users\Vaishu\Desktop\Work\project_03_ML\creditcard.csv")
print(df.head())
df.info()
df.describe()
'''
As per the count per column, we have no null values. Also, feature selection is 
not the case for this use case. 
'''
#%%%
df['Class'].value_counts()
'''

0=normal transanction, 1=fraudulant transanction
We have 284315 non fraud cases and 492 fraud class . The data is highly imbal
'''

#%% count of fraud and normal transanctions

nonfraud = df[df.Class==0]
fraud = df[df.Class==1]

print("\nAmount starts for non fraud class\n")
print(nonfraud.Amount.describe())
print("\nAmount starts for fraud class\n")
print(fraud.Amount.describe())
#%%
plt.figure(figsize = [12,6])
sns.kdeplot(df[(df.Class == 0) & (df.Amount < 3000)].Amount, label = 'Not Fraud')
sns.kdeplot(df[(df.Class == 1) & (df.Amount < 3000)].Amount, label = 'Fraud')
plt.legend(fontsize = 12)
'''
The graph clearly indicates the high imbalance in the data set
'''
#%%
plt.figure(figsize = [12,6])
sns.kdeplot(df[df.Class == 0].Time, label = 'Not Fraud')
sns.kdeplot(df[df.Class == 1].Time, label = 'Fraud')
plt.legend(fontsize = 12)
'''
We can see that time doesnt have much impact so we can delete the column

'''
#%%
df.drop(['Time'], axis=1, inplace=True)
df.shape
#%%
X = df.drop('Class', axis = 1).values
y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42,stratify=y)
y_train
y_test
#%%
classweights = class_weight.compute_class_weight('balanced',classes= np.unique(y_train),y= y_train)
classweights
classweights_dict=dict(zip(np.unique(y_train),classweights))
print("Class weight:",classweights_dict)
#%%
sc = StandardScaler()
amount = df['Amount'].values
df['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

#%%RandomizedSearchCV
#
n_estimators      = [int(x) for x in np.linspace(start = 200, stop = 250, num = 5)]

# Number of features to consider at every split
max_features      = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth         = [2,4,5]

# Minimum number of samples required to split a node
min_samples_split = [2,5]

# Minimum number  of samples required at each leaf node
min_samples_leaf  = [1, 2]

# Method of selecting samples for training each tree
bootstrap         = [True, False]

#Class weight for imbalance data
class_weight      =[classweights_dict]


#%%Defining Random Forest Model 
rf_Model = RandomForestClassifier()
#%%# Create the param grid

param_grid = {'n_estimators'      : n_estimators,
               'max_features'     : max_features,
               'max_depth'        : max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf' : min_samples_leaf,
               'bootstrap'        : bootstrap,
               'class_weight'     :class_weight}
print(param_grid)
#%%RandomizedSearchCv
rf_RandomGrid = RandomizedSearchCV(estimator = rf_Model, param_distributions = param_grid, cv = 10, verbose=2,scoring="recall", n_jobs = -1)
result=rf_RandomGrid.fit(X_train, y_train)

result.best_params_

print("\nBest Hyperparameters: %s\n" % result.best_params_)


best_random_grid=result.best_estimator_
best_random_grid.fit(X_train, y_train)

y_pred=best_random_grid.predict(X_test)

print("\n Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\nAccuracy Score: {}\n".format(accuracy_score(y_test,y_pred)))
print("Classification report: \n{}".format(classification_report(y_test,y_pred)))



###########################################################################################
#%% XGBoost Classifier
classifier=XGBClassifier()
#%% Defining the grid

param_grid = {
     "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
     "max_depth"        : [3, 4, 7, 10, 25],
     "min_child_weight" : [1, 3, 5, 7],
     "gamma"            : [0.0, 0.1, 0.2 , 0.3, 0.4],
     "colsample_bytree" : [0.3, 0.4, 0.5 , 0.7],
     "scale_pos_weight" : [1, 3, 5, 10, 25]
     }

print(param_grid)
#%% using HalvingGridSearch CV
halving_grid_search=HalvingGridSearchCV(estimator=classifier,param_grid=param_grid,cv=10,n_jobs=-1,scoring="recall",verbose=2)
result=halving_grid_search.fit(X_train,y_train)
result.best_params_

print("\nBest Hyperparameters: \n%s" % result.best_params_)

best_grid=result.best_params_
xgb_clf=XGBClassifier(**best_grid)
xgb_clf.fit(X_train, y_train)
y_pred=xgb_clf.predict(X_test)

print("\n Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\nAccuracy Score: {}\n".format(accuracy_score(y_test,y_pred)))
print("Classification report: \n{}".format(classification_report(y_test,y_pred)))

#%%

