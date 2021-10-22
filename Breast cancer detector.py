#!/usr/bin/env python
# coding: utf-8

# # BREAST CANCER CLASSIFICATION

# In[30]:


#Zeyad-elshaarawy 
10/22/2021


'''Predicting if the cancer diagnosis is benign or malignant based on several observations/features

#30 features are used, examples:

  - radius (mean of distances from center to points on the perimeter)
  - texture (standard deviation of gray-scale values)
  - perimeter
  - area
  - smoothness (local variation in radius lengths)
  - compactness (perimeter^2 / area - 1.0)
  - concavity (severity of concave portions of the contour)
  - concave points (number of concave portions of the contour)
  - symmetry 
  - fractal dimension ("coastline approximation" - 1)
Datasets are linearly separable using all 30 input features

Number of Instances: 569

Class Distribution: 212 Malignant, 357 Benign

Target class:

   - Malignant
   - Benign
   
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
'''


# In[53]:


#importing libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[54]:


# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# In[55]:


cancer


# In[56]:


cancer.keys()


# In[57]:


print(cancer['DESCR'])


# In[58]:


print(cancer['target_names'])


# In[59]:


print(cancer['target'])


# In[60]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


# In[61]:


df_cancer.head()


# In[62]:


df_cancer.tail()


# In[63]:


sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# In[64]:


plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 


# # Training the model 
# 

# In[65]:


#spliting the labeles from the features 
X = df_cancer.drop(['target'],axis=1)
X


# In[66]:


y = df_cancer['target']
y


# In[67]:


#spliting into training and testing 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)


# In[68]:


X_train.shape


# In[69]:


X_train.shape


# In[70]:


y_train.shape


# In[71]:


y_test.shape


# In[72]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)


# # Model evaluation 
# 
# 

# In[73]:


y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)


# In[74]:


sns.heatmap(cm, annot=True)


# In[75]:


print(classification_report(y_test, y_predict))


# # improving the model 

# In[76]:


min_train = X_train.min()
min_train


# In[77]:


range_train = (X_train - min_train).max()
range_train


# In[84]:


X_train_scaled = (X_train - min_train)/range_train
X_train_scaled


# In[85]:


sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)


# In[86]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[87]:


sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)


# In[88]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[89]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


# In[90]:


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm,annot=True,fmt="d")


# In[93]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


# In[94]:


from sklearn.model_selection import GridSearchCV


# In[95]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[96]:


grid.fit(X_train_scaled,y_train)


# In[97]:


grid.best_params_


# In[98]:


grid.best_estimator_


# In[99]:


grid_predictions = grid.predict(X_test_scaled)


# In[100]:


cm = confusion_matrix(y_test, grid_predictions)


# In[101]:


sns.heatmap(cm, annot=True)


# In[102]:


print(classification_report(y_test,grid_predictions))


# In[ ]:




