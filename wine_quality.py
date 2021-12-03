# -*- coding: utf-8 -*-
"""wine quality.ipynb


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from google.colab import files
uploaded= files.upload()

data= pd.read_csv("winequality-red.csv")

data

data.info()

data.isnull().sum()

data["quality"].value_counts()

counts= data["quality"].value_counts()
sns.barplot(x=counts.index,y=counts)
plt.xlabel('quality')
plt.ylabel('counts')

data['quality']= data['quality'].apply(lambda x:1 if x>=6 else 0)

data['quality'].nunique()

x= data.drop('quality',axis=1)
y= data['quality']

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=12)

x_train

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_train= scaler.fit_transform(x_train)
X_test= scaler.fit_transform(x_test)

X_train

def func(model):
  model.fit(X_train,y_train)
  pred= model.predict(X_test)
  acc= accuracy_score(pred,y_test)
  print("accuracy:{:.2f}%".format(acc*100))
  print(classification_report(y_test,pred))
  cf_matrix= confusion_matrix(y_test,pred)
  sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,fmt= '0.2%')
  return acc

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
acc_log= func(log_reg)

from sklearn.tree import DecisionTreeClassifier
model_dtc=DecisionTreeClassifier()
acc_dtc= func(model_dtc)
