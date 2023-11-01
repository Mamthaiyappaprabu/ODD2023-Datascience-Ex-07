# Ex-07-FEATURE SELECTION :
## AIM :
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## EXPLANATION :
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM :
### STEP 1 :
Read the given Data
### STEP 2 :
Clean the Data Set using Data Cleaning Process
### STEP 3 :
Apply Feature selection techniques to all the features of the data set
### STEP 4 :
Save the data to the file


## CODE :
### NAME : MAMTHA I
### REG NO : 212222230076
```
# DATA PREPROCESSING BEFORE FEATURE SELECTION:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()

#checking data
df.isnull().sum()

#removing unnecessary data variables
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()

#cleaning data
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()

#removing outliers 
plt.title("Dataset with outliers")
df.boxplot()
plt.show()

cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()

from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()

from sklearn.preprocessing import OrdinalEncoder
climate = ['male','female']
en= OrdinalEncoder(categories = [climate])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()

from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()

import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()

import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"] 

# FEATURE SELECTION:
# FILTER METHOD:
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()

# HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features

# BACKWARD ELIMINATION:
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
       cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

# RFE (RECURSIVE FEATURE ELIMINATION):
model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

# OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

# FINAL SET OF FEATURE:
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

# EMBEDDED METHOD:
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()

```
## OUTPUT :
### DATA PREPROCESSING BEFORE FEATURE SELECTION:
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/00c20afa-f66f-4148-8d00-6a870ae037fc)
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/4d312e9c-f88e-4c90-af80-095d1997765a)
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/ec725a93-f203-47a3-b138-4193215eb0aa)
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/5cd3c41e-32a8-43db-89e8-2cdf0bc9095a)
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/17ec9127-4d7a-4a18-a70b-4930b84a089e)
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/eaec3cc8-b2b6-4e5b-828d-5f54c22a81e1)
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/e990c8cd-a178-4132-9916-d80a285b9d4b)

### FEATURE SELECTION :
### FILTER METHOD :
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/0b8d3ec1-3c18-4923-9fb1-bd4e46b017e3)
### HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/deeac637-ded9-447b-9bb8-30e24224b454)

### WRAPPER METHOD :
### BACKWARD ELIMINATION:
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/38a8f1f3-73e9-4f1f-b4f3-38e78ab1e1ad)
## RFE (RECURSIVE FEATURE ELIMINATION):
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/012c0a7c-b9e8-46e3-98e6-2eb0325e0c2b)
## OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/159a43e2-20f9-4baf-8894-3c92bdb75e67)
## FINAL SET OF FEATURE:
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/164c4499-e07d-482d-b5d8-5c2ae06805b8)
## EMBEDDED METHOD:
![image](https://github.com/harinidq/Ex-07-Feature-Selection/assets/113497680/16afc0a8-b963-42a4-a834-1df346a089bf)
# RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.












