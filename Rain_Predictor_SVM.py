# -*- coding: utf-8 -*-
"""
@author: Roshan Joe Vincent
"""

#Importing Libraries
import pandas as pd

#Loading Data from CSV file
dataset = pd.read_csv('Weather.csv')

#Check if table has missing values
pd.isnull(dataset).any(1).nonzero()[0]

#Drop rows that have missing values
dataset.drop(pd.isnull(dataset).any(1).nonzero()[0], inplace = True)

#Breaking down Independent and Dependent variables
X = dataset.iloc[: , 1:4].values #upperbound is omitted
y = dataset.iloc[:, 4].values.astype(int)
y_temp = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# =============================================================================
# #Perform Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# =============================================================================

# =============================================================================#Perform min max Feature Scaling
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler(feature_range=(0,1))
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)


#Fitting the Classifier
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting rain/no
y_pred = classifier.predict(X_test)

#Analysing our model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=5)
scores.mean() 

#Analysing our model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



#Printing Results
print('Our model has correctly predicted that there will be no rain for', cm[0,0] , 'days.')
print('Our model has correctly predicted that there will be rain for', cm[1,1] , 'days.')



