# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

##Import All Relevant Packages

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import traceback
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


#Import Data
try:    
    df = pd.read_excel(r'C:\Assignment\Bankruptcy dataset.xlsx')
    print("\n Number of rows of data fetched is:", len(df))    
except Exception:
    traceback.print_exc()
#32,553

#Check df
df.head()

#Describe DF
desc = df.describe()  
print(df.groupby(['bankruptcy?']).size())

out_path = r'C:\Assignment\Data_Desc.xlsx'
writer = pd.ExcelWriter(out_path,engine='xlsxwriter')
desc.to_excel(writer)
writer.save()

#Check Data Duplicacy and Event Rate
df.count()
df['bankruptcy?'].sum()
df.duplicated().sum()
##Observation - Data Duplicates present. Make it unique

#Subset Dataframe without Duplicates to be used for Model Training
df2 = df.drop_duplicates()
#207 Duplicates 
df2.count()
#32,346
df2['bankruptcy?'].sum()

##Univariate Analysis 
##Correlation
cor = df2.corr()

out_path = r'C:\Assignment\corr.xlsx'
writer = pd.ExcelWriter(out_path,engine='xlsxwriter')
cor.to_excel(writer)
writer.save()

#Bivariate Analysis
##Bucketing all continuos variables to understand relation with Bankruptcy Rate as weight of evidence
##Plots
for i in range(1,65):
    b = [min(df2['Attr'+str(i)]),0,0.05,0.1,0.5,1,max(df2['Attr'+str(i)])]
    df2['Attr'+str(i)+'_grp'] = pd.cut(df2['Attr'+str(i)],bins=b, labels= False)
    temp = df2.groupby(['Attr'+str(i)+'_grp'])['bankruptcy?'].agg(['sum','count']).reset_index()
    temp['br'] = (temp['sum']/temp['count'])*100
# create visual
    fig, ax = plt.subplots(figsize=(12,8))
    plt.xlabel('Attribute Buckets')
    plt.ylabel('Bankruptcy Rate')
    plt.title('B Rate with Attribute Buckets')
    plt.bar(temp['Attr'+str(i)+'_grp'],temp['br'],alpha=.5)
    temp['count'].plot(secondary_y=True)
    plt.savefig('C://Assignment//Plots/Plot'+str(i))
    print("Plot for Attr saved")

##Check for Multicollinearity - VIF    
X = df.loc[:, df.columns != 'bankruptcy?']
y = df.loc[:, df.columns == 'bankruptcy?']

#Drop columns with more than 5% Nulls
#del X[['Attr21','Attr27','Attr37','Attr45','Attr60']
#del X['Attr37']

#Drop All rows with Nans in it
## Loss of Information to be made up by transformation or Imputation
X = X.dropna()
X.count()
#29,181

from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print(vif_data)

out_path = r'C:\Assignment\VIF.xlsx'
writer = pd.ExcelWriter(out_path,engine='xlsxwriter')
vif_data.to_excel(writer)
writer.save()

#Non Multi Collinear Variables 
raw_vars = [['Attr41','Attr5','Attr15','Attr55','Attr59','Attr57','Attr61']]
#Multicollinear Attributes
vect = df[['Attr6',	'Attr3',	'Attr64',	'Attr25',	'Attr24',	'Attr50',	'Attr45',	'Attr60',	'Attr53',	'Attr40',	'Attr1',	'Attr4',	'Attr32',	'Attr51',	'Attr46',	'Attr29',	'Attr38',	'Attr2',	'Attr10',	'Attr12',	'Attr62',	'Attr47',	'Attr52',	'Attr28',	'Attr54',	'Attr26',	'Attr30',	'Attr48',	'Attr34',	'Attr23',	'Attr31',	'Attr35',	'Attr16',	'Attr36',	'Attr9',	'Attr22',	'Attr33',	'Attr11',	'Attr63',	'Attr39',	'Attr58',	'Attr56',	'Attr8',	'Attr17',	'Attr19',	'Attr42',	'Attr49',	'Attr13',	'Attr20',	'Attr44',	'Attr43',	'Attr18']]
vect = vect.dropna()

##Data Preprocessing
sc = StandardScaler()
Sc_New =sc.fit_transform(vect)
# Applying PCA function on training# and testing set of X component
from sklearn.decomposition import PCA  
pca=PCA(n_components=10)
Sc_New=pca.fit_transform(Sc_New)
Sc_New= Sc_New.flatten()
#X_test=pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
explained_variance.sum()
##86% explained variance with 10 Principal Components when Nan removed ; 76% when Nan replaced with 0

#Stratified Sampling on Bankruptcy for Validation split
val = df2.groupby('bankruptcy?', group_keys=False).apply(lambda x: x.sample(frac=0.05))
##Remove Validation set from main dataframe
df2 = pd.concat([df2, val, val]).drop_duplicates(keep=False).reset_index(drop=True)

##Data Sampling for Model Training, Testing & Validation
X2 = df2.loc[:, df2.columns != 'bankruptcy?']
y2 = df2.loc[:, df2.columns == 'bankruptcy?']


#Generate Principal Components for Multicollinear Attributes  
vect2 = X2[['Attr6',	'Attr3',	'Attr64',	'Attr25',	'Attr24',	'Attr50',	'Attr45',	'Attr60',	'Attr53',	'Attr40',	'Attr1',	'Attr4',	'Attr32',	'Attr51',	'Attr46',	'Attr29',	'Attr38',	'Attr2',	'Attr10',	'Attr12',	'Attr62',	'Attr47',	'Attr52',	'Attr28',	'Attr54',	'Attr26',	'Attr30',	'Attr48',	'Attr34',	'Attr23',	'Attr31',	'Attr35',	'Attr16',	'Attr36',	'Attr9',	'Attr22',	'Attr33',	'Attr11',	'Attr63',	'Attr39',	'Attr58',	'Attr56',	'Attr8',	'Attr17',	'Attr19',	'Attr42',	'Attr49',	'Attr13',	'Attr20',	'Attr44',	'Attr43',	'Attr18']]
vect2 = np.nan_to_num(vect2)
PC2 = pca.transform(vect2)
PC2_df = pd.DataFrame(PC2)
#Sc2= Sc2.flatten()

X_Transform = pd.concat([X2[['Attr41','Attr5','Attr15','Attr55','Attr59','Attr57','Attr61']], PC2_df,y2], axis=1,ignore_index=False)
X_Transform.columns = X_Transform.columns.astype(str)
X_Transform.rename(columns = {'0':'PC0','1':'PC1', '2':'PC2','3':'PC3','4':'PC4', '5':'PC5','6':'PC6','7':'PC7', '8':'PC8','9':'PC9'}, inplace = True)
#30,729
#Drop All rows with Nans in it : 631 records removed
## Loss of Information to be made up by transformation or Imputation
X_Transform = X_Transform.dropna()
X_Transform.count()
#30,098

y2 = X_Transform.loc[:, X_Transform.columns == 'bankruptcy?']
X_Transform = X_Transform.loc[:, X_Transform.columns != 'bankruptcy?']



X_train, X_test, y_train, y_test = train_test_split(X_Transform, y2, test_size=0.2, random_state=0)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train.values.ravel())

y_train_pred = logreg.predict(X_train)
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))
#Accuracy of logistic regression classifier on train set: 0.95

##Predicting the test set results and calculating the accuracy
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
##Accuracy of logistic regression classifier on test set: 0.95

##Confusion Matrix
from sklearn.metrics import confusion_matrix
#confusion_matrix = confusion_matrix(y_test, y_pred)
#print(confusion_matrix)
cm_train = confusion_matrix(y_train, y_train_pred)
print(cm_train)


##Precision, Recall & FScore
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print(classification_report(y_train, y_train_pred))


##ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train.values.ravel())
  
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
  

##Precision, Recall & FScore
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print(classification_report(y_train, y_train_pred))

##Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
cm_train = confusion_matrix(y_train, y_train_pred)
print(cm_train)


##ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

##Random Forest with all given features
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

# using the feature importance variable
import pandas as pd
feature_imp = pd.Series(clf.feature_importances_, index = X_train.columns).sort_values(ascending = False)
feature_imp

#X2 = X2[['Attr46',	'Attr24',	'Attr58',	'Attr56',	'Attr34',	'Attr35',	'Attr39',	'Attr61',	'Attr40',	'Attr6',	'Attr44',	'Attr5',	'Attr26',	'Attr41',	'Attr25',	'Attr15',	'Attr55',	'Attr29',	'Attr20',	'Attr16',	'Attr47',	'Attr22',	'Attr3',	'Attr38',	'Attr42',	'Attr13',	'Attr18',	'Attr50',	'Attr48',	'Attr4',	'Attr54',	'Attr51',	'Attr9']]

##Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

##Precision, Recall & FScore
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print(classification_report(y_train, y_train_pred))

##Confusion Matrix
from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print(confusion_matrix)
cm_train = confusion_matrix(y_train, y_train_pred)
print(cm_train)
