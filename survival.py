# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:36:57 2021

@author: ludex
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Embarked'].fillna('S', inplace=True)
train['Age'].fillna(29, inplace=True)
test['Embarked'].fillna('S', inplace=True)
test['Age'].fillna(29, inplace=True)
test['Fare'].fillna(7.75, inplace=True)

sex_train = pd.get_dummies(train['Sex'], drop_first=True)
embarked_train = pd.get_dummies(train['Embarked'], drop_first=True)
sex_test = pd.get_dummies(test['Sex'], drop_first=True)
embarked_test = pd.get_dummies(test['Embarked'], drop_first=True)

train = pd.concat([train, sex_train, embarked_train], axis=1)
test = pd.concat([test, sex_test, embarked_test], axis=1)

train.drop(['PassengerId','Name','Ticket','Cabin','Sex','Embarked'], axis=1, inplace=True)
test.drop(['PassengerId','Name','Ticket','Cabin','Sex','Embarked'], axis=1, inplace=True)

x_train = train.iloc[:, 1: ].values
y_train = train.iloc[:, 0].values

x_test = test.iloc[:,:].values

standardscaler = StandardScaler()
x_train = standardscaler.fit_transform(x_train)
x_test = standardscaler.transform(x_test)

classifier = RandomForestClassifier(n_estimators=50, criterion='entropy',
                                    random_state=0, oob_score=True, n_jobs=-1)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from numpy import savetxt

savetxt('survival.csv',y_pred, delimiter=',')

#cm = confusion_matrix(y_test,y_pred)
#accuracy = accuracy_score(y_test, y_pred)
#
#fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#auc_score = auc(fpr, tpr)
#
#plt.plot(y_test, y_pred)
#plt.legend()
#plt.show()
#
##cross validation
#accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
#accuracies.mean()