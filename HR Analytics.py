#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# In[2]:


ds=pd.read_csv("E:/SOFTWARE/Datatrained/HR Analytics/ibm-hr-analytics-employee-attrition-performance/WA_Fn-UseC_-HR-Employee-Attrition.csv")
ds


# In[3]:


ds.info()


# In[4]:


ds.isnull().sum()


# In[5]:


sn.boxplot(data=ds['Age'])
plt.show()


# In[6]:


sn.boxplot(data=ds['DailyRate'])
plt.show()


# In[7]:


sn.boxplot(data=ds['DistanceFromHome'])
plt.show()


# In[8]:


sn.boxplot(data=ds['Education'])
plt.show()


# In[9]:


sn.boxplot(data=ds['EmployeeCount'])
plt.show()


# In[10]:


sn.boxplot(data=ds['EmployeeNumber'])
plt.show()


# In[11]:


sn.boxplot(data=ds['EnvironmentSatisfaction'])
plt.show()


# In[12]:


sn.boxplot(data=ds['HourlyRate'])
plt.show()


# In[13]:


sn.boxplot(data=ds['JobInvolvement'])
plt.show()


# In[14]:


sn.boxplot(data=ds['JobLevel'])
plt.show()


# In[15]:


sn.boxplot(data=ds['JobSatisfaction'])
plt.show()


# In[16]:


sn.boxplot(data=ds['MonthlyIncome'])
plt.show()


# In[17]:


sn.boxplot(data=ds['MonthlyRate'])
plt.show()


# In[18]:


sn.boxplot(data=ds['NumCompaniesWorked'])
plt.show()


# In[19]:


sn.boxplot(data=ds['PercentSalaryHike'])
plt.show()


# In[20]:


sn.boxplot(data=ds['PerformanceRating'])
plt.show()


# In[21]:


sn.boxplot(data=ds['RelationshipSatisfaction'])
plt.show()


# In[22]:


sn.boxplot(data=ds['StandardHours'])
plt.show()


# In[23]:


sn.boxplot(data=ds['StockOptionLevel'])
plt.show()


# In[24]:


sn.boxplot(data=ds['TotalWorkingYears'])
plt.show()


# In[25]:


sn.boxplot(data=ds['TrainingTimesLastYear'])
plt.show()


# In[26]:


sn.boxplot(data=ds['WorkLifeBalance'])
plt.show()


# In[27]:


sn.boxplot(data=ds['YearsAtCompany'])
plt.show()


# In[28]:


sn.boxplot(data=ds['YearsInCurrentRole'])
plt.show()


# In[29]:


sn.boxplot(data=ds['YearsSinceLastPromotion'])
plt.show()


# In[30]:


sn.boxplot(data=ds['YearsWithCurrManager'])
plt.show()


# In[31]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.Attrition=le.fit_transform(ds.Attrition)


# In[32]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.BusinessTravel=le.fit_transform(ds.BusinessTravel)


# In[33]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.Department=le.fit_transform(ds.Department)


# In[34]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.EducationField=le.fit_transform(ds.EducationField)


# In[35]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.JobRole=le.fit_transform(ds.JobRole)


# In[36]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.Gender=le.fit_transform(ds.Gender)


# In[37]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.JobRole=le.fit_transform(ds.JobRole)


# In[38]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.MaritalStatus=le.fit_transform(ds.MaritalStatus)


# In[39]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.Over18=le.fit_transform(ds.Over18)


# In[40]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.OverTime=le.fit_transform(ds.OverTime)


# In[41]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.OverTime=le.fit_transform(ds.OverTime)


# In[42]:


ds.info()


# In[ ]:





# In[ ]:





# #Removing outlier using Zscore
# from scipy.stats import zscore
# z=np.abs(zscore(ds))
# threshold=3
# new_ds=ds[(z<3)]

# In[43]:


ds.info()


# new_ds.info()

# In[44]:


#creating target variable and seperating data into X and Y
x=ds.drop("Attrition",axis=1)
y=ds['Attrition']

#Spliting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)


# In[45]:


#Since the Preprocessing and EDA is done we can build our model using Different Algorithms
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
y_pred


# In[46]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[47]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[48]:


#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rd=RandomForestClassifier(random_state=0)
rd.fit(x_train,y_train)
y_pred=rd.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[49]:


from sklearn.svm import SVC   
classifier = SVC(kernel='linear')  

classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[50]:


#We will try to improve performance of our best model using GridSearchCV
from sklearn.model_selection import GridSearchCV 
   
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 

grid.fit(x_train, y_train)

# print best parameter after tuning 
print(grid.best_params_) 

predictions = grid.predict(x_test)
print(classification_report(y_test, predictions))


# In[52]:


# Parameter GridSearchCV 
search_param = {
    "n_estimators": [100],
    "criterion": ["gini", "entropy"],
    'max_features': [0.5, 1.0, "sqrt"],
    'max_depth': [4, 5, 6, 7, 8, None],
}



#Hyper tunning Random Forest
forest = RandomForestClassifier(random_state = 0)
grid   = GridSearchCV(forest, search_param, cv = 12, verbose = 0)
grid.fit(x_train, y_train)

print(" Best Score :",grid.best_score_)
print(" Best Params :",grid.best_params_)


# In[53]:



predictions = grid.predict(x_test)
print(classification_report(y_test, predictions))


# We are getting best result after hyertuning RandomForestClassifier.
# Our best score is 86.

# In[54]:


import pickle
filename = 'HR Analytics.pkl'
pickle.dump(grid,open(filename, 'wb'))


# In[ ]:




