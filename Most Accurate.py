# %% Import all the necessary:

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# %% Function:

def rater(val, delimiter: str = "/"):
    if type(val) != type(np.nan):
        n, d = list(map(np.float64, val.split(delimiter)))
        return n / d


# %% Importing the data and getting the basic infos:

data = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/zomato.csv")
original = data.copy()
print(data.info())
stats = data.describe()
print(data.shape)
print(data.isna().sum())
print(data.duplicated().sum())

# %% Removing Unnecessary Columns: [By Logical]

data.drop(['url', 'address', 'reviews_list', 'menu_item', 'phone', 'dish_liked', 'rest_type', 'cuisines'], axis=1,
          inplace=True)
print(data.info())

# %% Making miscellaneous data normal:

# Making NEW values in rate column as Nan values:

print('The NEW values: ')
print(data.isin(['NEW']).sum())
data.replace('NEW', np.nan, inplace=True)
print('The NEW values: ')
print(data.isin(['NEW']).sum())

# %%
# Making rate column as numerical from it is miscellaneous form:

data['rate'].astype(object)
data['rate'].replace('nan', np.nan, inplace=True)
data['rate'].replace('-', np.nan, inplace=True)
outlier = data[data['rate'].isna()]
outlier = outlier[outlier['votes'] != 0]
outlier = outlier.iloc[:, 3:5]
data.drop(outlier.index, inplace=True)
data.reset_index(drop=True)
data['rate'] = data['rate'].apply(rater)
data['rate'].replace(np.nan, 0, inplace=True)

# %%
# Making approx cost to numerical form:

data['approx_cost(for two people)'] = data['approx_cost(for two people)'].astype(str)
data['approx_cost(for two people)'] = data['approx_cost(for two people)'].str.replace(',', '')
data['approx_cost(for two people)'].astype(object)
data['approx_cost(for two people)'].replace('nan', np.nan, inplace=True)

# %% Performing operations for the remaining null values:

impute_data = data.isna().sum()
impute_data = impute_data[impute_data != 0]
print(impute_data)

# In this, we have 2 columns of categorical and 1 continuously:

impute_data = impute_data
Impute = SimpleImputer()
data['approx_cost(for two people)'] = Impute.fit_transform(
    data['approx_cost(for two people)'].to_numpy().reshape((-1, 1)))
print(data.isna().sum())

# Dropping missing values for categorical as they of less number:

print(data.isna().sum())
print(data.shape)
data.dropna(subset=['location'], inplace=True)
print(data.isna().sum())

# %%
# region Anomaly Detection and Outliers Detection NEED TO LEARN!!

stats = data.describe()
q1 = stats.loc['25%', :]
q3 = stats.loc['75%', :]
iqr = q3 - q1
outlier_rate = data[(data['rate'] < q1['rate'] - (1.5 * iqr['rate'])) | (data['rate'] > q3['rate'] + (1.5 * iqr['rate']))]
print(outlier_rate.shape)
data.drop(outlier_rate.index, inplace=True)
data.reset_index(drop=True)
print(data.shape)

# %% Splitting Data between Independent and Dependent:

X = data[['online_order', 'book_table', 'votes', 'rate', 'listed_in(type)', 'listed_in(city)', 'location']]
Y = data['approx_cost(for two people)'].values

# %% Standardization of continuous data:

sc = MinMaxScaler()
X.iloc[:, 2:4] = sc.fit_transform(X.iloc[:, 2:4])
Y = sc.fit_transform(Y.reshape((-1, 1)))

# %% Converting Categorical Values of Yes or No to Label:

feature = LabelEncoder()
X.loc[:, 'online_order'] = feature.fit_transform(X.loc[:, 'online_order'])
X.loc[:, 'book_table'] = feature.fit_transform(X.loc[:, 'book_table'])

# %% Converting Categorical Values into Numerical ones:

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [4, 5, 6])], remainder="passthrough")
X_trans = (ct.fit_transform(X).toarray())

# %% Splitting the data into train and test data:

X_train, X_test, Y_train, Y_test = train_test_split(X_trans, Y, test_size=0.2)

# %% Using Linear Model to fit the data:

lr = LinearRegression()
lr.fit(X_train, Y_train)

# %% Predict the values:

Y_predict = lr.predict(X_test)

# %% Calculating the Accuracy of the analysis:

lr = r2_score(Y_test, Y_predict)

# %%
# Preparing Extra Tree Regression

RF_Model = RandomForestRegressor(n_estimators=650, random_state=245, min_samples_leaf=.0001, verbose=2, n_jobs=-1)
RF_Model.fit(X_train, Y_train.ravel())
Y_predict = RF_Model.predict(X_test)
rf = r2_score(Y_test, Y_predict)

# %%

ET_Model = ExtraTreesRegressor(n_estimators=120, verbose=2, n_jobs=-1)
ET_Model.fit(X_train, Y_train)
Y_predict = ET_Model.predict(X_test)
et = r2_score(Y_test, Y_predict)

# %%

xgb_r = xg.XGBRegressor(n_estimators=1000, verbosity=3, n_jobs=-1)
xgb_r.fit(X_train, Y_train)
Y_predict = xgb_r.predict(X_test)
xbr = r2_score(Y_test, Y_predict)

# %%

print("The R2 score of Linear Regression: ", lr)
print("The R2 score of Random Forest: ", rf)
print("The R2 score of Extra Tree: ", et)
print("The R2 score of XGB Regression: ", xbr)

# %%
