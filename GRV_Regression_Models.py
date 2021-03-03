#!/usr/bin/env python
# coding: utf-8

# In[33]:


get_ipython().system('pip install catboost')


# In[34]:


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt


# In[3]:


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    return summary


# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True) 


# In[ ]:


get_ipython().system('ls "/content/drive/My Drive"')


# # New Section

# In[35]:


df = pd.read_csv("Thinkful Data Science Projects/Datasets/samp_gravity.csv",sep=",")


# In[ ]:


df.value_counts()


# In[ ]:


df.head()


# In[ ]:


print ("Total number of rows in dataset = {}".format(df.shape[0]))
print ("Total number of columns in dataset = {}".format(df.shape[1]))


# In[ ]:


df2=df.dropna()


# In[ ]:


print ("Total number of rows in dataset = {}".format(df2.shape[0]))
print ("Total number of columns in dataset = {}".format(df2.shape[1]))


# In[ ]:


result = resumetable(df)
result


# In[ ]:


df.dtypes


# In[ ]:


import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

import missingno as msno


# In[ ]:


msno.bar(df)


# In[ ]:


# Find how much data is missing in each column
# Function to look at missing rows per column
def missing(dataset):
    columns = dataset.columns
    print('MISSING ROWS per COLUMN')
    for column in columns:
        percentage = (dataset[column].isnull().sum() / len(df)) * 100
        print('{}: {}, {:0.2f}%'.format(column, dataset[column].isnull().sum(), percentage))
        
        
# Missing rows per column       
missing(df)


# In[ ]:


corr = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
s = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


plt.show()
corr


# In[36]:


# Drop all columns with more than 50% of it's values missing
df_dc = df[df.columns[df.isnull().mean() < 0.5]]
df_dc.head()


# In[7]:


df_num=df_dc.drop(['iso_o','iso_d','iso2_o','iso2_d'],axis=1)
df_num.head()


# In[37]:


for col in  df_num.columns:
    df_num[col] = pd.to_numeric(df_num[col], errors='coerce')


# In[ ]:


df_num.dtypes


# In[9]:


df_num["gdp_o"] = pd.to_numeric(df_num["gdp_o"])


# In[10]:


# credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction. 
# One of the best notebooks on getting started with a ML problem.

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[11]:


df_num_missing= missing_values_table(df_num)
df_num_missing


# ## Detecting missing data visually using Missingno library

# ## Visualizing the locations of the missing data

# In[ ]:


msno.matrix(df_num)


# In[ ]:


df_num.dtypes


# In[ ]:


df_num


# ## SimpleImputer

# In[38]:


## SimpleImputer
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
data_with_imputed_values = my_imputer.fit_transform(df_num)


# In[ ]:


data_with_imputed_values


# In[39]:


new_data = pd.DataFrame(data_with_imputed_values,columns=df_num.columns)


# In[ ]:


new_data.isnull().sum()


# In[15]:


new_data.head()


# In[ ]:





# In[ ]:


# Find how much data is missing in each column
# Function to look at missing rows per column
def missing(dataset):
    columns = dataset.columns
    print('MISSING ROWS per COLUMN')
    for column in columns:
        percentage = (dataset[column].isnull().sum() / len(new_data)) * 100
        print('{}: {}, {:0.2f}%'.format(column, dataset[column].isnull().sum(), percentage))
        
        
# Missing rows per column       
missing(new_data)


# In[ ]:


new_data.dtypes


# ## Catboost

# In[40]:


# fraction of rows 
  
# here you get 20 % row from the df 
# make put into another dataframe df1 
test = new_data.sample(frac =.2) 
  
# Now select 80 % rows from df1 
train=new_data.sample(frac =.80) 


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[41]:


train_df=train
test_df = test


# In[42]:


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score


# In[43]:


#Creating a training set for modeling and validation set to check model performance
X = train.drop(['gdp_o'], axis=1)
y = train.gdp_o

from sklearn.model_selection import train_test_split
X_train, X_vaidation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)


# In[25]:


#Look at the data type of variables
X.dtypes


# In[44]:


categorical_features_indices = np.where(X.dtypes != np.float)[0]


# In[45]:


#importing library and building model
from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=1000, depth=3, learning_rate=0.1, loss_function='RMSE')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_vaidation, y_validation),plot=True)


# In[ ]:


submission = pd.DataFrame()
submission['distw'] = test['distw']
submission['gdp_o'] = test['gdp_o']
submission['distw'] = model.predict(test)
submission.to_csv("Submission.csv")


# In[ ]:


submission


# In[ ]:


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


# In[ ]:


submission.plot.scatter(x='distw',y='gdp_o')


# ##  Regression Models

# ## Random_Forest_Regression_using_Scikit_Learn.ipynb

# In[ ]:





# In[ ]:


#import libraries for pre-processing
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

from dateutil.parser import parse
from datetime import datetime
from scipy.stats import norm

# import all what you need for machine learning
import sklearn
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state= 42)


# In[ ]:


#fit decision tree
tree = DecisionTreeRegressor()
tree.fit(x_train, y_train)
#fit random forest
forest = RandomForestRegressor(n_jobs=-1)
forest.fit(x_train, y_train)
#fit regression
lin_reg = LinearRegression(n_jobs=-1)
lin_reg.fit(x_train, y_train)


# In[ ]:


models= [('lin_reg', lin_reg), ('random forest', forest), ('decision tree', tree), ('Catboost', model)]
from sklearn.metrics import mean_squared_error
for i, model in models:    
    predictions = model.predict(x_train)
    MSE = mean_squared_error(y_train, predictions)
    RMSE = np.sqrt(MSE)
    msg = "%s = %.2f" % (i, round(RMSE, 2))
    print('RMSE of', msg)


# In[ ]:


for i, model in models:
    # Make predictions on train data
    predictions = model.predict(x_train)
    # Performance metrics
    errors = abs(predictions - y_train)
    # Calculate mean absolute percentage error (MAPE)
    mape = np.mean(100 * (errors / y_train))
    # Calculate and display accuracy
    accuracy = 100 - mape    
    #print result
    msg = "%s= %.2f"% (i, round(accuracy, 2))
    print('Accuracy of', msg,'%')


# In[ ]:


models= [('lin_reg', lin_reg), ('forest', forest), ('dt', tree),('Catboost', model)]
scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

#for each model I want to test three different scoring metrics. Therefore, results[0] will be lin_reg x MSE, 
# results[1] lin_reg x MSE and so on until results [8], where we stored dt x r2

results= []
metric= []
for name, model in models:
    for i in scoring:
        scores = cross_validate(model, x_train, y_train, scoring=i, cv=5, return_train_score=True)
        print(scores)
        results.append(scores)


# In[ ]:


#THIS IS FOR Linear regression
#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy
LR_RMSE_mean = np.sqrt(-results[0]['test_score'].mean())
LR_RMSE_std= results[0]['test_score'].std()
# note that also here I changed the sign, as the result is originally a negative number for ease of computation
LR_MAE_mean = -results[1]['test_score'].mean()
LR_MAE_std= results[1]['test_score'].std()
LR_r2_mean = results[2]['test_score'].mean()
LR_r2_std = results[2]['test_score'].std()

#THIS IS FOR RF
RF_RMSE_mean = np.sqrt(-results[3]['test_score'].mean())
RF_RMSE_std= results[3]['test_score'].std()
RF_MAE_mean = -results[4]['test_score'].mean()
RF_MAE_std= results[4]['test_score'].std()
RF_r2_mean = results[5]['test_score'].mean()
RF_r2_std = results[5]['test_score'].std()

#THIS IS FOR DT
DT_RMSE_mean = np.sqrt(-results[6]['test_score'].mean())
DT_RMSE_std= results[6]['test_score'].std()
DT_MAE_mean = -results[7]['test_score'].mean()
DT_MAE_std= results[7]['test_score'].std()
DT_r2_mean = results[8]['test_score'].mean()
DT_r2_std = results[8]['test_score'].std()


#THIS IS FOR CB
CB_RMSE_mean = np.sqrt(-results[9]['test_score'].mean())
CB_RMSE_std= results[9]['test_score'].std()
CB_MAE_mean = -results[10]['test_score'].mean()
CB_MAE_std= results[10]['test_score'].std()
CB_r2_mean = results[11]['test_score'].mean()
CB_r2_std = results[11]['test_score'].std()


# In[ ]:


modelDF = pd.DataFrame({
    'Model'       : ['Linear Regression', 'Random Forest', 'Decision Trees','Catboost'],
    'RMSE_mean'    : [LR_RMSE_mean, RF_RMSE_mean, DT_RMSE_mean, CB_RMSE_mean],
    'RMSE_std'    : [LR_RMSE_std, RF_RMSE_std, DT_RMSE_std, CB_RMSE_std],
    'MAE_mean'   : [LR_MAE_mean, RF_MAE_mean, DT_MAE_mean, CB_MAE_mean],
    'MAE_std'   : [LR_MAE_std, RF_MAE_std, DT_MAE_std,CB_MAE_std],
    'r2_mean'      : [LR_r2_mean, RF_r2_mean, DT_r2_mean,CB_r2_mean],
    'r2_std'      : [LR_r2_std, RF_r2_std, DT_r2_std,CB_r2_std],
    }, columns = ['Model', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std', 'r2_mean', 'r2_std'])

modelDF.sort_values(by='r2_mean', ascending=False)


# 

# In[28]:


from catboost import Pool

train_data = Pool(data=X_train,
                  label=y_train,
                  )

valid_data = Pool(data=X_vaidation,
                  label=y_validation,
                  )


# In[29]:


get_ipython().system('pip install shap')


# In[30]:


from multiprocessing import Pool


# In[31]:


import shap
explainer = shap.TreeExplainer(model) # insert your model
shap_values = explainer.shap_values(train_data) # insert your train Pool object

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[:100,:], X_train.iloc[:100,:])


# In[32]:


shap.summary_plot(shap_values, X_train)

