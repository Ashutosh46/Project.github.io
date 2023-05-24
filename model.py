## First we import the necessary libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
## to display all the columns of the dataset 
pd.set_option("display.max_columns",None)
## we import our dataset
dataset=pd.read_csv("F:\DATA SETS\CustomerlifetimeValue-copy.csv")

## we check the shape of the data,i.e, the number of rows and columns
dataset.shape

## to check the dimension of the data
dataset.ndim

## to print the top 5 records of the data
dataset.head()

## to check whether the data has any null values
dataset.isnull().sum()

## we find the number of numerical features present in the dataset
numerical_features=[feature for feature in dataset.columns if dataset[feature].dtypes!="O"]
print("count of numerical features:",len(numerical_features))

## we see the content of the numerical features
dataset[numerical_features].head()

## we create a separate variable for the temporal variable
temp_var=dataset["Effective To Date"]
temp_var

## We try to find if there is any relationship between the temporal variable and the target variable
## We observe that our target variable(CLV) have gone through cyclical fluctuations 
dataset.groupby("Effective To Date")["Customer Lifetime Value"].median().plot()
plt.xlabel("Effective To Date")
plt.ylabel("median customer lifetime value")
plt.title("date vs clv")
plt.show()

## We find the number of discrete numerical features in the dataset
discrete_features=[feature for feature in numerical_features if len(dataset[feature].unique())<20]
print("count of discrete variables :",len(discrete_features))

## We print the name of the discrete features
print(discrete_features)

## We print the data in the discrete features 
dataset[discrete_features].head()

## We plot a barplot to analyze the relationship between the list of discrete features and median_CLV
for feature in discrete_features:
    dataset.groupby(feature)["Customer Lifetime Value"].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel("median CLV")
    plt.show()
	
## We find the number of continous numerical features in the dataset
continous_features=[feature for feature in numerical_features if feature not in discrete_features]
print("count of continous features:",len(continous_features))

## We print the name of the continous features
print(continous_features)

## We print the data in continous features
dataset[continous_features].head()

## We plot histograms to find the distribution of the continous features
for feature in continous_features:
    plt.hist(dataset[feature],bins=20)
    plt.xlabel(feature)
    plt.ylabel("count")
    plt.title(feature)
    plt.show()
	
## We observe that the continous features are skewed so we apply logarithmic function to transform them
## only those features are transformed who does not have 0 as a value 
for feature in continous_features:
    if 0 in dataset[feature].unique():
        pass
    else:
        dataset[feature]=np.log(dataset[feature])
        plt.hist(dataset[feature],bins=20)
        plt.xlabel(feature)
        plt.ylabel("count")
        plt.title(feature)
        plt.show()
	
## We make boxplots to find out the presence of outliers in the continous features
for feature in continous_features:
    sns.boxplot(y=dataset[feature])
    plt.xlabel(feature)
    plt.show()

## We find the list of categorical features present in the dataset
categorical_features=[feature for feature in dataset.columns if dataset[feature].dtypes=="O"]
categorical_features

## We remove the "Customer" column as it is irrelevant
categorical_features.remove("Customer")

## We remove "Effective To Date" column as it is irrelevant to treat it as a categorical feature
categorical_features.remove("Effective To Date")

## Now we print the updated list of categorical features
categorical_features

## We find the number of categorical features present in the dataset
len(categorical_features)

## We print the data in the categorical features
dataset[categorical_features].head()

## We find the cardinality of the categorical features,i.e, the number of sub-categories present in each categorical feature
for feature in categorical_features:
    print("feature is {} and number of sub-categories are {}".format(feature,len(dataset[feature].unique())))
	
## We plot barplot to observe the relationship between the categorical features and the median of the target variable(CLV)
for feature in categorical_features:
    dataset.groupby(feature)["Customer Lifetime Value"].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel("median Customer Lifetime Value")
    plt.show()
	
## We try to find the percentage of missing values present in the categorical features
categorical_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes=="O"]
for feature in categorical_nan:
    print("{} has {} % missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))
	
## We fill the nan values in the categorical features with a label named "missing"
dataset[categorical_nan]=dataset[categorical_nan].fillna("missing")
## after replacing the nan values we check the whether there is any nan value present
dataset[categorical_nan].isnull().sum()

## We print the data of the categorical features which had nan values earlier
dataset[categorical_nan].head()

## We try to find the percentage of missing values present in the numerical features
numerical_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!="O"]
for feature in numerical_nan:
    print("{} has {} % missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))
	
## as we have observed outliers in the data points of numercial features so we replace the nan values with it's median value
for feature in numerical_nan:
    median_value=dataset[feature].median()
    dataset[feature].fillna(median_value,inplace=True)
dataset[numerical_nan].isnull().sum()

## We print the dataset to find out check whether the nan values have been by median or not
dataset.head()

## We replace the sub categories which are present in less than 10% of the dataset with "rare_var"label 
for feature in categorical_features:
    temp=dataset.groupby(feature)["Customer Lifetime Value"].count()/len(dataset)
    temp_df=temp[temp>0.01].index
    dataset[feature]=np.where(dataset[feature].isin(temp_df),dataset[feature],"rare_var")
	
## We print the dataset to observe the change
dataset.head(20)

## We drop the unnecessary features from the dataset
dataset=dataset.drop(["Customer","Effective To Date"],axis=1)

## We perform one hot encoding on categorical variables
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
columns_to_one_hot = ['State','Response','Coverage','Education','EmploymentStatus','Gender','Location Code','Marital Status','Policy Type','Policy','Renew Offer Type','Sales Channel','Vehicle Class','Vehicle Size']
encoded_array = enc.fit_transform(dataset.loc[:,columns_to_one_hot])
dataset_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
dataset_sklearn_encoded = pd.concat([dataset,dataset_encoded],axis=1)
dataset_sklearn_encoded.drop(labels= columns_to_one_hot,axis=1,inplace=True)

## We print the encoded dataset
dataset_sklearn_encoded

## We separate the independent and dependent variables from the dataset
x=dataset_sklearn_encoded.drop("Customer Lifetime Value",axis=1)
y=dataset_sklearn_encoded["Customer Lifetime Value"]

## We print the independent varibale
x

## We print the dependent variable
y=pd.DataFrame(y)
y

## We split our data into train and test data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=18)

## We use Standardization method to scale down all the features in the dataset 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
y_train=scaler.fit_transform(y_train)
y_test=scaler.transform(y_test)

## We print the transformed x_train
x_train

## We print the transformed y_train
y_train

## We import certain modules from the sklearn library
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

## We fit our dataset into lasso regression to select relevant features of the dataset to be considered for modelling
feature_sel_model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
feature_sel_model.fit(x,y)

## We print an array indicating which features are selected
feature_sel_model.get_support()

## We count the number of selected features  
selected_feat=x.columns[(feature_sel_model.get_support())]
print("number of features selected: {}".format(len(selected_feat)))

## We create a list of the selected features and print them
selected_feat=list(selected_feat)
print(selected_feat)

## We create a list of independent variables and build the x_train dataframe with it
feature_scale=[feature for feature in dataset_sklearn_encoded.columns if feature not in ["Customer Lifetime Value"]]
x_train=pd.DataFrame(x_train,columns=feature_scale)
x_train

x_test=pd.DataFrame(x_test,columns=feature_scale)
x_test

## Now we reduce the dimension of x_train data by considering only the selected features
x_train=x_train[['Monthly Premium Auto', 'Months Since Last Claim', 'Months Since Policy Inception', 'Number of Open Complaints', 'Number of Policies', 'Total Claim Amount', 'EmploymentStatus_Employed', 'Marital Status_Single', 'Renew Offer Type_Offer1', 'Renew Offer Type_Offer2', 'Vehicle Class_Four-Door Car', 'Vehicle Class_SUV']]

## Now we reduce the dimension of x_test data by considering only the selected features
x_test=x_test[['Monthly Premium Auto', 'Months Since Last Claim', 'Months Since Policy Inception', 'Number of Open Complaints', 'Number of Policies', 'Total Claim Amount', 'EmploymentStatus_Employed', 'Marital Status_Single', 'Renew Offer Type_Offer1', 'Renew Offer Type_Offer2', 'Vehicle Class_Four-Door Car', 'Vehicle Class_SUV']]

## We convert the x_train dataframe into an array
x_train=np.array(x_train)
x_train

## We convert the x_test dataframe into an array
x_test=np.array(x_test)
x_test

y_train

y_test

## We check the shape of the x_train data
x_train.shape

## We check the shape of the x_test data
x_test.shape

## We check the shape of the y_train data
y_train.shape

## We check the shape of the y_test data
y_test.shape

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators = 500,random_state=6)
regressor.fit(x_train,y_train)

## We predict using the x_test data
y_pred=regressor.predict(x_test)

## We print the predicted values
print(y_pred)

y_test[1]

x_test[1]

print(regressor.predict([x_test[1]]))

print(regressor.predict([[100,3,12,0,1,1200,1,1,1,1,0,0]]))

## We construct a dataframe to display the actual and predicted values together
df=pd.DataFrame({"actual":y_test.ravel(),"predicted":y_pred.ravel()})

## We print the dataframe
df

## ## We use score to see how well the data is performing for train data
regressor.score(x_train,y_train)

## We use score to see how well the data is performing for test data
regressor.score(x_test,y_test)

## We check the r2_score 
metrics.r2_score(y_test,y_pred)

metrics.mean_squared_error(y_test,y_pred)

from sklearn.model_selection import learning_curve

x=x[['Monthly Premium Auto', 'Months Since Last Claim', 'Months Since Policy Inception', 'Number of Open Complaints', 'Number of Policies', 'Total Claim Amount', 'EmploymentStatus_Employed', 'Marital Status_Single', 'Renew Offer Type_Offer1', 'Renew Offer Type_Offer2', 'Vehicle Class_Four-Door Car', 'Vehicle Class_SUV']]

import pickle 

pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))