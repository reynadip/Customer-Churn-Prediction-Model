# %%
#Data preprocessing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt #for visualization 


import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory


#Read the dataset
data_df = pd.read_csv("customer_churn_large_dataset.csv",na_values=['id'])

#Get overview of the data
def dataoveriew(df, message):
    print(f'{message}:n')
    print('Number of rows: ', df.shape[0])
    print("nNumber of features:", df.shape[1])
    print("nData Features:")
    print(df.columns.tolist())
    print("nMissing values:", df.isnull().sum().values.sum())
    print("nUnique values:")
    print(df.nunique())

dataoveriew(data_df, 'Overview of the dataset')



# %%
data_df.sample(4)

# %%
data_df.drop('CustomerID',axis='columns',inplace=True)
data_df.drop('Name',axis='columns',inplace=True)

# %%
data_df.dtypes

# %%
data_df.Gender.values

# %%
data_df[pd.to_numeric(data_df.Subscription_Length_Months,errors='coerce').isnull()]

# %%
data_df[pd.to_numeric(data_df.Monthly_Bill,errors='coerce').isnull()]


# %%
data_df[pd.to_numeric(data_df.Total_Usage_GB).isnull()]

# %%
Churn_leaving=data_df[data_df.Churn==1].Subscription_Length_Months
Churn_not_leaving=data_df[data_df.Churn==0].Subscription_Length_Months

plt.xlabel("Subscription_Length_Months")
plt.ylabel("Number of customers")
plt.title("Customer Churn Prediction Vis")

plt.hist([Churn_leaving,Churn_not_leaving],color=['red','green'],label=["Churn_leaving",'Churn_not_leaving'])
plt.legend()

# %%
Churn_leaving=data_df[data_df.Churn==1].Monthly_Bill
Churn_not_leaving=data_df[data_df.Churn==0].Monthly_Bill
plt.xlabel("Monthly_Bill")
plt.ylabel("Number of customers")
plt.title("Customer Churn Prediction Vis")

plt.hist([Churn_leaving,Churn_not_leaving],color=['red','green'],label=["Churn_leaving",'Churn_not_leaving'])
plt.legend()

# %%
Churn_leaving=data_df[data_df.Churn==1].Total_Usage_GB
Churn_not_leaving=data_df[data_df.Churn==0].Total_Usage_GB
plt.xlabel("Total_Usage_GB")
plt.ylabel("Number of customers")
plt.title("Customer Churn Prediction Vis")

plt.hist([Churn_leaving,Churn_not_leaving],color=['red','green'],label=["Churn_leaving",'Churn_not_leaving'])
plt.legend()

# %%
#feature engneering

def print_unique_col_values(df):
    for column in data_df:
        if data_df[column].dtypes=='object':
            print(f'{column}:{data_df[column].unique()}')

# %%
print_unique_col_values(data_df)

# %%
data_df['Gender'].replace({'Female':1,'Male':0},inplace=True)

# %%
data_df['Gender'].unique()

# %%
#one hot encoding


df2=pd.get_dummies(data=data_df,columns=['Location'])
df2.columns

# %%
df2.sample(4)

# %%
df2.dtypes

# %%
#feature scaling

cols_to_scale=['Subscription_Length_Months','Monthly_Bill','Total_Usage_GB']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])


# %%
df2.sample(4)

# %%
X = df2.drop('Churn',axis = 'columns')
y = df2['Churn']

# %%
#splitting test train datasets


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=5)

# %%
X_train.shape

# %%
X_test.shape

# %%
X_train[:10]

# %%
len(X_train.columns)

# %%
#model_building

import tensorflow as tf
from tensorflow import keras


model=keras.Sequential([
    keras.layers.Dense(10, input_dim=(10), activation='relu'),
    keras.layers.Dense(3,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid'),
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit(X_train,y_train,epochs=100)

# %%
model.evaluate(X_test,y_test)

# %%
yp=model.predict(X_test)
yp[:5]

# %%
y_test[:5]

# %%
y_pred=[]
for element in yp:
    if element >0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

y_pred[:10]

# %%
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_pred))

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%
def modeling(alg, alg_name, params={}):
    model = alg(**params) #Instantiating the algorithm class and unpacking parameters if any
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #Performance evaluation
    def print_scores(alg, y_true, y_pred):
        print(alg_name)
        acc_score = accuracy_score(y_true, y_pred)
        print("accuracy: ",acc_score)
        pre_score = precision_score(y_true, y_pred)
        print("precision: ",pre_score)
        rec_score = recall_score(y_true, y_pred)
        print("recall: ",rec_score)
        f_score = f1_score(y_true, y_pred, average='weighted')
        print("f1_score: ",f_score)

    print_scores(alg, y_test, y_pred)
    return model

# Running logistic regression model
log_model = modeling(LogisticRegression, 'Logistic Regression')

# %%
#optimization_of_features

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
log = LogisticRegression()
rfecv = RFECV(estimator=log, cv=StratifiedKFold(10, random_state=50, shuffle=True), scoring="accuracy")
rfecv.fit(X, y)

# %%
X_rfe = X.iloc[:, rfecv.support_]

# %%
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
plt.grid()
plt.xticks(range(1, X.shape[1]+1))
plt.xlabel("Number of Selected Features")
plt.ylabel("CV Score")
plt.title("Recursive Feature Elimination (RFE)")
plt.show()

print("The optimal number of features: {}".format(rfecv.n_features_))

# %%
#Saving dataframe with optimal features
X_rfe = X.iloc[:, rfecv.support_]


# %%
# Running logistic regression model
log_model = modeling(LogisticRegression, 'Logistic Regression')

# %%
#Random forest
rf_model = modeling(RandomForestClassifier, "Random Forest Classification")

# %%
##Hyperparameter tuning
# define model
model = LogisticRegression()

# define evaluation
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space
from scipy.stats import loguniform
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 1000)

# define search
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)

# execute search
result = search.fit(X_rfe, y)
# summarize result
# print('Best Score: %s' % result.best_score_)
# print('Best Hyperparameters: %s' % result.best_params_)
params = result.best_params_

#Improving the Logistic Regression model
log_model = modeling(LogisticRegression, 'Logistic Regression Classification', params=params)

# %% [markdown]
# 


