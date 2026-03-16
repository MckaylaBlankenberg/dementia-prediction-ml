#Question 1.1
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.feature_selection import RFE
#from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("dementia.csv")

df = df.iloc[:,1:]
print(df.head())

print(df.isnull().sum())
print(df.head())

df['EducationLevel'] = df['EducationLevel'].replace('None', np.nan)

common = df["EducationLevel"].mode()[0]
df["EducationLevel"].fillna(common, inplace=True)

print(df.isnull().sum())
print(df.head())

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Diagnosis", axis=1)
y = df['Diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data preprocessing complete")
print("Number of featurs after encoding:", X.shape[1])

#Question 1.2.a
from sklearn.feature_selection import SelectKBest, f_classif

select = SelectKBest(score_func=f_classif, k=10)
fit = select.fit(X,y)

select_features_kbest = X.columns[select.get_support()]
print("Top selected features using SelectKBest: ")
print(select_features_kbest)

#Question 1.2.b
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)

print("Top features selected by RFE: ")
print(X.columns[rfe.support_])


