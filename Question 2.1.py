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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Question 2.1.a. Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_kbest = X_train[select_features_kbest]
X_test_kbest = X_test[select_features_kbest]

log_model_kbest = LogisticRegression(random_state=42)
log_model_kbest.fit(X_train_kbest, y_train)

y_pred_log_model_kbest = log_model_kbest.predict(X_test_kbest)

print("*********Logistic Regression - SelectKBest***********")
print("Accuracy: ", accuracy_score(y_test, y_pred_log_model_kbest))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred_log_model_kbest))
print("Classification Report: ", classification_report(y_test, y_pred_log_model_kbest))

rfe_features = X.columns[rfe.support_]
X_train_rfe = X_train[rfe_features]
X_test_rfe = X_test[rfe_features]

log_rfe = LogisticRegression(random_state=42)
log_rfe.fit(X_train_rfe, y_train)

y_pred_log_rfe = log_rfe.predict(X_test_rfe)

print("\n*********Logistic Regression - RFE***********")
print("Accuracy: ", accuracy_score(y_test, y_pred_log_rfe))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred_log_rfe))
print("Classification Report: ", classification_report(y_test, y_pred_log_rfe))

#Question 2.1.b. Decision Tree
X_train_kbest = X_train[select_features_kbest]
X_test_kbest = X_test[select_features_kbest]

Dtree_kbest = DecisionTreeClassifier(random_state=42)
Dtree_kbest.fit(X_train_kbest, y_train)
y_pred_Dtree_kbest = Dtree_kbest.predict(X_test_kbest)

print("************Decision Tree - SelectKBest*********")
print("Accuracy: ", accuracy_score(y_test, y_pred_Dtree_kbest))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred_Dtree_kbest))
print("\nClassification Report: ", classification_report(y_test, y_pred_Dtree_kbest))

rfe_features = X.columns[rfe.support_]
X_train_rfe = X_train[rfe_features]
X_test_rfe = X_test[rfe_features]

Dtree_rfe = DecisionTreeClassifier(random_state=42)
Dtree_rfe.fit(X_train_rfe, y_train)

y_pred_Dtree_rfe = Dtree_rfe.predict(X_test_rfe)

print("************Decision Tree - RFE*********")
print("Accuracy: ", accuracy_score(y_test, y_pred_Dtree_rfe))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred_Dtree_rfe))
print("\nClassification Report: ", classification_report(y_test, y_pred_Dtree_rfe))



