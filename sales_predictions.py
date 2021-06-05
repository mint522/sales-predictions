import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

filename = '/Users/jiali/Documents/Python_CodingDojo/sales_predictions.csv'
df = pd.read_csv(filename)
print(df.info())

#Missing values in 'Item_Weight'
#1463 values missing in 'Item_Weight' are all from year 1985
print(df[df['Item_Weight'].isna()])
print(df[df['Item_Weight'].isna()]['Outlet_Establishment_Year'].value_counts())
print(df.corr().loc['Item_Outlet_Sales', 'Item_Weight'])
df.drop(columns=['Item_Weight'], inplace=True)

#Missing values in 'Outlet_Size'
#'Supermarket Type1' and 'Grocery' are missing 'Outlet_Type'
print(df[df['Outlet_Size'].isna()]['Outlet_Type'].value_counts())
print(df[(df['Outlet_Type']=='Supermarket Type1')]['Outlet_Size'].value_counts(dropna=False))
print(df[(df['Outlet_Type']=='Supermarket Type1')& (df['Outlet_Location_Type']=='Tier 1')]['Outlet_Size'].value_counts(dropna=False))
print(df[(df['Outlet_Type']=='Supermarket Type1')& (df['Outlet_Location_Type']=='Tier 2')]['Outlet_Size'].value_counts(dropna=False))
print(df[(df['Outlet_Type']=='Supermarket Type1')& (df['Outlet_Location_Type']=='Tier 3')]['Outlet_Size'].value_counts(dropna=False))
print(df[(df['Outlet_Type']=='Grocery Store')]['Outlet_Size'].value_counts(dropna=False))

#Histogram of sales in outlet types
plt.figure(1)
df.loc[df['Outlet_Type']=='Supermarket Type1']['Item_Outlet_Sales'].hist(bins=20, edgecolor='black', alpha=0.7, label='Supermarket Type1')
df.loc[df['Outlet_Type']=='Supermarket Type2']['Item_Outlet_Sales'].hist(bins=5, edgecolor='black', alpha=0.8, label='Supermarket Type2')
df.loc[df['Outlet_Type']=='Supermarket Type3']['Item_Outlet_Sales'].hist(bins=50, edgecolor='black', alpha=0.7, label='Supermarket Type3')
df.loc[df['Outlet_Type']=='Grocery Store']['Item_Outlet_Sales'].hist(bins=12, edgecolor='black', alpha=0.5, label='Grocery Store')
plt.legend(loc='upper right')
plt.xlabel('Sales')
plt.ylabel('Count')
plt.title('Sales distribution of different outlet types')

#Histogram of sales in outlet location types
plt.figure(2)
df.loc[df['Outlet_Location_Type']=='Tier 3']['Item_Outlet_Sales'].hist(bins=8, edgecolor='black', alpha=0.7, label='Tier 3')
df.loc[df['Outlet_Location_Type']=='Tier 2']['Item_Outlet_Sales'].hist(bins=9, edgecolor='black', alpha=0.7, label='Tier 2')
df.loc[df['Outlet_Location_Type']=='Tier 1']['Item_Outlet_Sales'].hist(bins=20, edgecolor='black', alpha=0.5, label='Tier 1')
plt.legend(loc='upper right')
plt.xlabel('Sales')
plt.ylabel('Count')
plt.title('Sales distribution of different outlet location types')

#Histogram of sales in outlet sizes
plt.figure(3)
df.loc[df['Outlet_Size']=='Small']['Item_Outlet_Sales'].hist(bins=8, edgecolor='black', alpha=0.7, label='Small')
df.loc[df['Outlet_Size']=='Medium']['Item_Outlet_Sales'].hist(bins=25, edgecolor='black', alpha=0.7, label='Medium')
df.loc[df['Outlet_Size']=='High']['Item_Outlet_Sales'].hist(bins=12, edgecolor='black', alpha=0.7, label='High')
plt.legend(loc='upper right')
plt.xlabel('Sales')
plt.ylabel('Count')
plt.title('Sales distribution of different outlet sizes')

#Check correlation between features
#Encoding Categorical Data
print(df['Outlet_Size'].value_counts())
df['Outlet_Size_Label'] = df['Outlet_Size'].replace({'Small': 0, 'Medium':1, 'High':2})
print(df['Item_Fat_Content'].value_counts())
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
df['Item_Fat_Content_Label'] = df['Item_Fat_Content'].replace({'Low Fat':0, 'Regular':1})
print(df['Outlet_Location_Type'].value_counts())
df['Outlet_Location_Type_Label'] = df['Outlet_Location_Type'].replace({'Tier 1':1, 'Tier 2':2, 'Tier 3': 3})
# Check correlation with existing data - delete rows with NaNs and check the correlation
# 'Outlet_Size' has low correlation with sales
# 'Outlet_Establishment_Year' has low correlation with sales
df_test = df.dropna(how='any')
print(df_test.corr().loc['Item_Outlet_Sales', :])
plt.figure(4)
sns.heatmap(df_test.corr(), annot=True, square=True, cmap='YlGnBu', xticklabels=(['Item Visibility', 'Item MRP', 'Outlet Est Year', 'Sales', 'Outlet Size', 'Item Fat', 'Outlet Loc']), yticklabels=(['Item Visibility', 'Item MRP', 'Outlet Est Year', 'Sales', 'Outlet Size', 'Item Fat', 'Outlet Loc']))
plt.title('Correlation between features')
df.drop(columns=['Outlet_Size', 'Outlet_Size_Label'], inplace=True)

# Outlet_Establishment_Year' vs Sales
outlet_sales = df.groupby(['Outlet_Identifier', 'Outlet_Establishment_Year'])[['Item_Outlet_Sales']].sum().sort_values('Item_Outlet_Sales')
print(outlet_sales)
outlet_sales.plot(kind='barh', legend=None)
plt.figure(5)
plt.xlabel('Sales')
plt.ticklabel_format(style='plain', axis='x')
plt.ylabel('Outlet ID, Est Year')
plt.title('Sales by outlet')
df.drop(columns=['Outlet_Establishment_Year'], inplace=True)

# MRP vs Sales
plt.figure(6)
plt.scatter(df['Item_MRP'], df['Item_Outlet_Sales'])
plt.xlabel('Item MRP')
plt.ylabel('Item Outlet Sales')
plt.title('Item MRP vs Sales')

# Outlet Location vs Sales
plt.figure(7)
tier1 = df.loc[df['Outlet_Location_Type']=='Tier 1', 'Item_Outlet_Sales']
tier2 = df.loc[df['Outlet_Location_Type']=='Tier 2', 'Item_Outlet_Sales']
tier3 = df.loc[df['Outlet_Location_Type']=='Tier 3', 'Item_Outlet_Sales']
plt.boxplot([tier1, tier2, tier3], labels=['Tier 1', 'Tier 2', 'Tier 3'], notch=True)
plt.ylabel('Item Outlet Sales')
plt.xlabel('Outlet Location Type')
plt.title('Outlet location vs Sales')

# Sales by outlet locaiton type
outlet_loc = df.groupby(['Outlet_Location_Type'])[['Item_Outlet_Sales']].sum().sort_values('Item_Outlet_Sales')
outlet_loc.plot(kind='barh', legend=None)
plt.figure(8)
plt.xlabel('Sales')
plt.ylabel('Outlet Location Type')
plt.ticklabel_format(style='plain', axis='x')
plt.title('Sales by outlet locaiton type')

# Outlet Type vs Sales
outlet_type = df.groupby(['Outlet_Type'])[['Item_Outlet_Sales']].sum().sort_values('Item_Outlet_Sales')
outlet_type.plot(kind='barh', legend=None)
plt.figure(9)
plt.xlabel('Sales')
plt.ylabel('Outlet Type')
plt.ticklabel_format(style='plain', axis='x')
plt.title('Sales by outlet type')

# Outlet Type and Location vs Sales
outlet_type_loc = df.groupby(['Outlet_Type', 'Outlet_Location_Type'])[['Item_Outlet_Sales']].sum().sort_values('Item_Outlet_Sales')
outlet_type_loc.plot(kind='barh', legend=None)
plt.figure(10)
plt.xlabel('Sales')
plt.ylabel('Outlet Type, Outlet Location Type')
plt.ticklabel_format(style='plain', axis='x')
plt.title('Sales by outlet type and location')

# Visibility vs Sales
plt.figure(11)
plt.scatter(df['Item_Visibility'], df['Item_Outlet_Sales'])
plt.xlabel('Item Visibility')
plt.ylabel('Item Outlet Sales')
plt.title('Item Visibility vs Sales')

#Encoding Categorical Data
#reference: https://colab.research.google.com/drive/1ocNoCkcyVzC_QipOOCa6kssFhABswEH8
print(df['Item_Type'].value_counts())
print(df['Outlet_Type'].value_counts())
df_dummies = pd.get_dummies(df, columns = ['Item_Type', 'Outlet_Type'], drop_first=True)
df_analyze = df_dummies.drop(columns=['Item_Identifier', 'Item_Fat_Content', 'Outlet_Identifier', 'Outlet_Location_Type'])
print(df_analyze.info())

X = df_analyze.drop(columns=['Item_Outlet_Sales'])
y = df_analyze['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

#Linear Regression
reg = LinearRegression(fit_intercept=True)
reg.fit(X_train, y_train)

print(reg.predict(X_train))
print(reg.predict(X_test))
print(reg.score(X_train, y_train))
print(reg.score(X_test, y_test))
#Mean squared error (MSE)
print(mean_squared_error(y_test, reg.predict(X_test)))
#Mean absolute error (MAE)
print(mean_absolute_error(y_test, reg.predict(X_test)))
#Root mean squared error (RMSE)
print(np.sqrt(mean_squared_error(y_test, reg.predict(X_test))))


#Random forest regression
estimator_range = [1] + list(range(10, 110, 10))
scores = {}
for estimator in estimator_range:
    random_reg = RandomForestRegressor(n_estimators=estimator, random_state=1, bootstrap=True)
    random_reg.fit(X_train, y_train)
    scores[estimator] = random_reg.score(X_test, y_test)
print(max(scores, key=scores.get))

random_reg = RandomForestRegressor(n_estimators=80)
random_reg.fit(X_train, y_train)
print(random_reg.predict(X_train))
print(random_reg.predict(X_test))
print(random_reg.score(X_train, y_train))
print(random_reg.score(X_test, y_test))
print(np.sqrt(mean_squared_error(y_test, random_reg.predict(X_test))))

# reference: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
print(random_reg.feature_importances_)
plt.figure(12)
plt.barh(X.columns, random_reg.feature_importances_)
plt.title('Feature Importances')


#KNN
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

estimator_range = list(range(1,20))
scores = {}
for estimator in estimator_range:
    random_reg = KNeighborsRegressor(n_neighbors=estimator)
    random_reg.fit(X_train, y_train)
    scores[estimator] = random_reg.score(X_test, y_test)
print(max(scores, key=scores.get))

knn = KNeighborsRegressor(n_neighbors=13)
knn.fit(X_train, y_train)
print(knn.predict(X_train))
print(knn.predict(X_test))
print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))
print(np.sqrt(mean_squared_error(y_test, knn.predict(X_test))))

plt.show()