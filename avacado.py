#!/usr/bin/env python
# coding: utf-8

# In[98]:



import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# In[99]:


ds=pd.read_csv("E:/SOFTWARE/Datatrained/Evaluation/avacado.csv")
ds


# In[100]:


ds.isnull().sum()


# In[101]:


ds.info()


# 
# We need to handle Date column by seperating year,month and day in seperate columns

# In[102]:


ds.describe()


# In[103]:


ds['year']=pd.DatetimeIndex(ds['Date']).year
ds['month']=pd.DatetimeIndex(ds['Date']).month
ds['day']=pd.DatetimeIndex(ds['Date']).day


# In[104]:


ds.drop(['Date'],inplace=True,axis=1)


# In[105]:


ds


# In[106]:


ds.info()


# In[107]:


g1 = ds.groupby(["region"]).size()
g1


# In[108]:


corr=ds.corr()
plt.figure(figsize=(16,5))
sn.heatmap(corr,annot=True,linewidths=.8)
plt.show()


# using correlation heatmap we can observe
# 1.Unnamed:0,AveragePrice,Large Bags and 4225 are highly correlated.
# 2.year is least correlated

# In[109]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.type=le.fit_transform(ds.type)


# In[110]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.region=le.fit_transform(ds.region)


# Converting categorical columns to numerical using LabelEncoder

# In[111]:


ds.info()


# In[112]:


sn.boxplot(data=ds["Unnamed: 0"])
plt.show()


# In[113]:


sn.boxplot(data=ds["Total Volume"])
plt.show()


# In[114]:


sn.boxplot(data=ds["4046"])
plt.show()


# In[115]:


sn.boxplot(data=ds["4225"])
plt.show()


# In[116]:


sn.boxplot(data=ds["4770"])
plt.show()


# In[117]:


sn.boxplot(data=ds["Total Bags"])
plt.show()


# In[118]:


sn.boxplot(data=ds["Small Bags"])
plt.show()


# In[119]:


sn.boxplot(data=ds["XLarge Bags"])
plt.show()


# In[120]:


plt.hist(ds['XLarge Bags'])


# Columns like "Total Volume,4046,4225,4770,total Bags,Small Bags,XLarge Bags" have outliers.
# We will try to remove outliers using zscore

# In[121]:


plt.show()


# In[122]:


sn.distplot(ds['XLarge Bags'])
plt.show()


# In[123]:


sn.distplot(ds['Unnamed: 0'])
plt.show()


# #Removing outlier using Zscore
# from scipy.stats import zscore
# z=np.abs(zscore(ds))
# threshold=3
# new_ds=ds[(z<3)]

# In[124]:


#creating target variable and seperating data into X and Y
x=ds.drop("region",axis=1)
y=ds['region']


# new_ds.info()

# In[125]:


ds.info()


# In[ ]:





# In[126]:


#
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)


# In[127]:


#Scling data using standardscalar method
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaledx=sc.fit_transform(x)


# In[128]:


#Since the Preprocessing and EDA is done we can build our model using Different Algorithms
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[129]:


y_pred=lr.predict(x_test)
y_pred


# In[130]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[131]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[132]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
rd=RandomForestClassifier(n_estimators=100)
rd.fit(x_train,y_train)
y_pred=rd.predict(x_test)


# In[133]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[134]:


from sklearn.svm import SVC   
classifier = SVC(kernel='linear')  

classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)


# In[135]:


y_pred = classifier.predict(x_test)
r2score=r2_score(y_test,y_pred)
print(f"Accuracu is {r2score*100}")


# In[136]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[137]:


#We will try to improve performance of our best model using GridSearchCV
from sklearn.model_selection import GridSearchCV 
   
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 

grid.fit(x_train, y_train)


# In[138]:


# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)


# In[139]:


predictions = grid.predict(x_test)
print(classification_report(y_test, predictions))


# Regression

# In[140]:


x=new_ds.drop("AveragePrice",axis=1)
y=new_ds['AveragePrice']


# In[141]:


ds.info()


# In[142]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[143]:


from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(x_train,y_train)


# In[144]:


y_pred = lin.predict(x_test)


# In[145]:


from sklearn.metrics import mean_squared_error, r2_score
R2 =r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
lin.score(x_test,y_test)
print('R2: ',R2)
print('RMSE: ',rmse)
print('Score: ',score)


# In[ ]:





# In[146]:



from sklearn.ensemble import RandomForestRegressor
rd=RandomForestRegressor(n_estimators = 100, random_state = 42)
rd.fit(x_train,y_train)
y_pred=rd.predict(x_test)


# In[147]:



R2 =r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
score=rd.score(x_test,y_test)
print('R2: ',R2)
print('RMSE: ',rmse)
print('Score: ',score)


# In[148]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train,y_train)
svr_y_pred=rd.predict(x_test)


# In[149]:



R2 =r2_score(y_test, svr_y_pred)
rmse = mean_squared_error(y_test, svr_y_pred)
score=rd.score(x_test,svr_y_pred)
print('R2: ',R2)
print('RMSE: ',rmse)
print('Score: ',score)


#  Model Saving

# In[150]:


#Creating pickle file
import pickle
filename='avacado.pkl'
pickle.dump(regressor,open(filename,'wb'))


# Conclusion

# In[151]:


a=np.array(y_test)
a


# In[152]:


predicted=np.array(regressor.predict(x_test))
predicted


# In[153]:


con=pd.DataFrame({"Original" : a,"Predicted":predicted},index=range(len(a)))
con


# In[ ]:




