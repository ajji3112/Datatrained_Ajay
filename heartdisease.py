#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# In[2]:


ds=pd.read_csv("E:/SOFTWARE/1.Datatrained/Evaluation/heartdisease.csv",na_values = "?")
ds


# In[3]:



ds=ds.rename(columns={"63":"age","1":"sex","4":"cp","140":"trestbps","260":"chol","0":"fbs","1.1":"restecg","112":"thalach","1.2":"exang","3":"oldpeak","2":"slope","?":"ca","?.1":"thal","2.1":"num"})
ds


# In[4]:


ds.isnull().sum()


# In[5]:


ds.nunique()


# In[6]:


ds['ca'].unique()


# In[7]:


ds['slope'].unique()


# In[8]:


ds['thal'].unique()


# In[9]:


ds['exang'].unique()


# In[10]:


ds['fbs'].unique()


# In[11]:


ds['num'].unique()


# In[12]:


ds.describe()


# In[13]:


ds.info()


# In[ ]:





# In[14]:


ds


# In[15]:


ds['slope'].fillna(ds['slope'].mode()[0], inplace = True)


# In[16]:


ds['oldpeak'].fillna(ds['oldpeak'].median(), inplace = True)


# In[17]:


ds['trestbps'].fillna(ds['trestbps'].median(), inplace = True)


# In[18]:


ds['chol'].fillna(ds['chol'].median(), inplace = True)


# In[19]:


ds['thalach'].fillna(ds['thalach'].median(), inplace = True)


# In[20]:


ds['fbs'].fillna(ds['fbs'].mode()[0], inplace = True)


# In[21]:


ds['exang'].fillna(ds['exang'].mode()[0], inplace = True)


# In[22]:


ds['slope'].fillna(ds['slope'].mode()[0], inplace = True)


# In[23]:


#Removing ca column since null values are more than 50%
ds.drop('ca',axis=1,inplace=True)


# In[24]:


#Removing thal column since null values are more than 50%
ds.drop('thal',axis=1,inplace=True)


# In[25]:


#Removing thal column since null values are more than 50%
ds.drop('chol',axis=1,inplace=True)


# In[26]:


#Removing thal column since null values are more than 50%
ds.drop('fbs',axis=1,inplace=True)


# In[27]:


#Removing thal column since null values are more than 50%
ds.drop('restecg',axis=1,inplace=True)


# In[28]:


corr=ds.corr()
plt.figure(figsize=(16,5))
sn.heatmap(corr,annot=True,linewidths=.8)
plt.show()


# In[29]:


ds.isnull().sum()


# In[30]:


sn.boxplot(data=ds['exang'])
plt.show()


# In[31]:


sn.boxplot(data=ds['oldpeak'])
plt.show()


# In[32]:


sn.boxplot(data=ds['thalach'])
plt.show()


# In[33]:


sn.boxplot(data=ds['trestbps'])
plt.show()


# In[34]:


from scipy.stats import skew


# In[35]:


for col in ds:
    print(col)
    print(skew(ds[col]))
    
    plt.figure()
    sn.distplot(ds[col])
    plt.show()


# In[36]:


ds_n=ds


# In[37]:


from scipy import stats
import numpy as np
z = np.abs(stats.zscore(ds_n))
print(z)

threshold = 3
print(np.where(z > 3))

ds_n = ds_n[(z < 3).all(axis=1)]


# In[38]:


ds.shape


# In[39]:


ds_n.shape


# In[ ]:





# In[40]:


#creating target variable and seperating data into X and Y
x=ds_n.drop("num",axis=1)
y=ds_n['num']


# In[ ]:





# #Scling data using standardscalar method
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# scaledx=sc.fit_transform(x)

# In[41]:


#
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)


# In[42]:


#Since the Preprocessing and EDA is done we can build our model using Different Algorithms
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[43]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
rd=RandomForestClassifier(n_estimators=100)
rd.fit(x_train,y_train)
y_pred=rd.predict(x_test)

print(classification_report(y_test, y_pred))

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[44]:


from sklearn.svm import SVC   
classifier = SVC(kernel='linear')  

classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


print(classification_report(y_test, y_pred))
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[45]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


print(classification_report(y_test, y_pred))
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[46]:


#We will try to improve performance of our best model using GridSearchCV
from sklearn.model_selection import GridSearchCV 
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


y_pred = classifier.predict(x_test)

print(" Best Score :",grid.best_score_)
print(" Best Params :",grid.best_params_)

 
predictions = grid.predict(x_test)
print(classification_report(y_test, predictions))


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:




