import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

filename = '/Users/jiali/Documents/Python_CodingDojo/sales_predictions.csv'
df = pd.read_csv(filename)
print(df.info())


#data clean
#delete column 'Item_Weight' and 'Outlet_Size'
df.drop(columns=['Item_Weight', 'Outlet_Size'], inplace=True)
#standardize
print(df['Item_Fat_Content'].value_counts())
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})


#Encoding Categorical Data
#referece: https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63
fat_label = LabelEncoder().fit_transform(df['Item_Fat_Content'])
df['Item_Fat_Content_Label'] = fat_label

print(df['Item_Type'].value_counts())
item_type_label = LabelEncoder().fit_transform(df['Item_Type'])
df['Item_Type_Label'] = item_type_label

print(df['Outlet_Location_Type'].value_counts())
outlet_location_type_label = LabelEncoder().fit_transform(df['Outlet_Location_Type'])
df['Outlet_Location_Type_Label'] = outlet_location_type_label

print(df['Outlet_Type'].value_counts())
outlet_type_label = LabelEncoder().fit_transform(df['Outlet_Type'])
df['Outlet_Type_Label'] = outlet_type_label

print(df.head(10))
print(df.info())


#Linear Regression
X = df[['Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year', 'Item_Fat_Content_Label', 'Item_Type_Label', 'Outlet_Location_Type_Label', 'Outlet_Type_Label']]
y = df['Item_Outlet_Sales']

reg = LinearRegression(fit_intercept=True)
reg.fit(X, y)
#Coefficient of Determination
print(r2_score(y, reg.predict(X)))
#Mean squared error (MSE)
print(mean_squared_error(y, reg.predict(X)))
#Mean absolute error (MAE)
print(mean_absolute_error(y, reg.predict(X)))
#Root mean squared error (RMSE)
print(np.sqrt(mean_squared_error(y, reg.predict(X))))


#KNN
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
knn = KNeighborsRegressor()
knn.fit(X, y)
print(knn.predict(X))
print(knn.score(X, y))


#visualization
item = df.groupby(['Item_Identifier'])[['Item_Outlet_Sales']].sum()
item_top_sales = item.sort_values('Item_Outlet_Sales', ascending=False).head(10)

outlet = df.groupby(['Outlet_Identifier'])[['Item_Outlet_Sales']].sum()
outlet_top_sales = outlet.sort_values('Item_Outlet_Sales', ascending=False).head(10)
print(outlet_top_sales)

#plots
plt.figure(figsize=(14,7))
plt.suptitle('Sales Predictions')
plt.subplots_adjust(hspace=0.5, wspace=0.3)

#subplot1
plt.subplot(2,3,1)
plt.bar(item_top_sales.index, item_top_sales['Item_Outlet_Sales'])
plt.xticks(fontsize=7, rotation=45)
plt.yticks(fontsize=7)
plt.xlabel('Item ID', fontsize=8)
plt.ylabel('Sales', fontsize=8)
plt.title('Top 10 sales by items', fontsize=10)

#subplot2
plt.subplot(2,3,2)
plt.bar(outlet_top_sales.index, outlet_top_sales['Item_Outlet_Sales'])
plt.xticks(fontsize=7, rotation=45)
plt.yticks(fontsize=7)
plt.xlabel('Outlet ID', fontsize=8)
plt.ylabel('Sales', fontsize=8)
plt.title('Top 10 sales by outlets', fontsize=10)

#histogram
plt.subplot(2,3,3)
df['Item_Outlet_Sales'].hist(bins=20)
plt.ticklabel_format(useOffset=False)
plt.xlabel('Item Outlet Sales', fontsize=8)
plt.ylabel('Count', fontsize=8)
plt.title('Item Outlet Sales Distrbutions', fontsize=10)

#boxplot
low_fat = df.loc[df['Item_Fat_Content']=='Low Fat', 'Item_Outlet_Sales']
regular = df.loc[df['Item_Fat_Content']=='Regular', 'Item_Outlet_Sales']
plt.subplot(2,3,4)
plt.boxplot([low_fat, regular], labels=['Low Fat', 'Regular'], notch=True)
plt.ylabel('Item Outlet Sales', fontsize=8)
plt.xlabel('Item Fat Content', fontsize=8)
plt.title('Item Outlet Sales by Item Fat Content', fontsize=10)

#heatmap
#reference: https://www.geeksforgeeks.org/how-to-change-the-colorbar-size-of-a-seaborn-heatmap-figure-in-python/
plt.subplot(2,3,6)
sns.heatmap(df.corr(), annot=None, square=True, cmap='Blues', xticklabels=(['Item Visibility', 'MRP', 'Outlet Est Year', 'Sales', 'Item Fat', 'Item Type', 'Outlet Loc', 'Outlet Type']), yticklabels=(['Item Visibility', 'MRP', 'Outlet Est Year', 'Sales', 'Item Fat', 'Item Type', 'Outlet Loc', 'Outlet Type']))
plt.yticks(rotation=0)
plt.title('Correlation between features', fontsize=10)
plt.show()

#Is there anything you can do to improve your model?
#Could remove records where 'Item_Visibility'==0 

#Which features are most associated with higher predicted sales?
#Item Outlet Sales is more related with Item MRP and Outlet Type