# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 13:06:01 2026

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 19:05:23 2026

@author: HP
"""

import numpy as np
import pandas as pd

titanic_dataset=pd.read_csv("C:/Users/HP/Desktop/Titanic-Dataset.csv")

#=======================EDA================================================
titanic_dataset.shape
titanic_dataset.columns
titanic_dataset.head(10)
titanic_dataset.info()

print(titanic_dataset.describe())

titanic_dataset.isnull().sum()


print(titanic_dataset["Survived"].value_counts())
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
titanic_dataset['Survived'].value_counts().plot(kind='bar')
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()


print(titanic_dataset['Pclass'].value_counts())
plt.figure(figsize=(6,4))
titanic_dataset['Pclass'].value_counts().sort_index().plot(kind='bar')
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.countplot(x='Pclass', hue='Survived', data=titanic_dataset)
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()


print(titanic_dataset['Sex'].value_counts())
plt.figure(figsize=(5,4))

sns.countplot(x='Sex', data=titanic_dataset)

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()


pd.crosstab(titanic_dataset['Sex'],titanic_dataset['Survived'])
plt.figure(figsize=(8,5))
sns.countplot(x='Sex',hue='Survived',data=titanic_dataset)
plt.xlabel("Gender")
plt.ylabel("Counts of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

#========================== Missing Values Handling ============================
titanic_dataset.isnull().sum()
titanic_dataset['Age'].median()

titanic_dataset['Age'].fillna(titanic_dataset['Age'].median(),inplace=True)
titanic_dataset['Embarked'].fillna(titanic_dataset['Embarked'].mode()[0],inplace=True)
titanic_dataset.drop('Cabin',axis=1,inplace=True)

titanic_dataset.isnull().sum()

titanic_dataset['Sex']=titanic_dataset['Sex'].map({
    'male':0,
    'female':1
    })
print(titanic_dataset['Sex'].head())
titanic_dataset['Embarked'].unique()

titanic_dataset= pd.get_dummies(titanic_dataset, columns=['Embarked'], drop_first=True)
titanic_dataset.columns

#=================== OUTLIERS DETECTION =============================

plt.figure(figsize=(10,5))
sns.histplot(titanic_dataset['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


titanic_dataset['Age'].describe()
plt.figure(figsize=(8,2))
sns.boxplot(x=titanic_dataset['Age'])
plt.title("Boxplot of Age")
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(titanic_dataset['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

titanic_dataset['Fare'].describe()
plt.figure(figsize=(8,2))
sns.boxplot(x=titanic_dataset['Fare'])
plt.title("Boxplot of Fare")
plt.show()

plt.figure(figsize=(6,2))
sns.boxplot(x=titanic_dataset['SibSp'])
plt.title("Boxplot of SibSp")
plt.show()

#============IQR METHOD =========================
Q1 = titanic_dataset['SibSp'].quantile(0.25)
Q3 = titanic_dataset['SibSp'].quantile(0.75)
IQR = Q3 - Q1

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)
outliers = titanic_dataset[(titanic_dataset['SibSp'] < lower_limit) | (titanic_dataset['SibSp'] > upper_limit)]
print(outliers)
print("Number of Outliers:", len(outliers))
outliers[['SibSp', 'Parch', 'Survived']]

# =========================FEATURE SELECTION =========================

titanic_dataset.drop('PassengerId',axis=1,inplace=True)
titanic_dataset.drop('Name', axis=1, inplace=True)
titanic_dataset.drop('Ticket', axis=1, inplace=True)
titanic_dataset.columns #fatured are selected 

#========================Model Training==================
x = titanic_dataset.drop('Survived', axis=1)
y = titanic_dataset['Survived']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

manual_input = [[1, 1, 30, 1, 0, 80, 0, 1]]

y_pred=model.predict(x_test)

#print(y_pred)


#========================Model Evaluation ======================
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

#================Confusion_matrix ================
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


#===================ROC Curve and AUC===========================================
from sklearn.metrics import roc_curve, roc_auc_score

y_prob = model.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(8,6))

plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.2f}')

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#====================================Classification Report================================
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))