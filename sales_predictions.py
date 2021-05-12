import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

filename = '/Users/jiali/Documents/Python_CodingDojo/sales_predictions.csv'
df = pd.read_csv(filename)

#delete column 'Item_Weight'
#keeping column 'Outlet_Size' for now, still deciding whether to delete the column or not. Can't delete NaN rows since too many records.
df.drop(columns='Item_Weight', inplace=True)

#data clean
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})


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
sns.heatmap(df.corr(), annot=None, square=True, cmap='Blues', xticklabels=(['Visibility', 'MRP', 'Year', 'Sales']), yticklabels=(['Visibility', 'MRP', 'Year', 'Sales']))
plt.yticks(rotation=0)
plt.title('Correction between features', fontsize=10)
plt.show()


#backup groups for future use
# item_fat_content = df.groupby(df['Item_Fat_Content'])['Item_Outlet_Sales'].sum()
# item_type = df.groupby(df['Item_Type'])['Item_Outlet_Sales'].sum()
# outlet_year = df.groupby(['Outlet_Establishment_Year'])['Item_Outlet_Sales'].sum()
# outlet_loc = df.groupby(['Outlet_Location_Type'])['Item_Outlet_Sales'].sum()
# outlet_type = df.groupby(['Outlet_Type'])['Item_Outlet_Sales'].sum()