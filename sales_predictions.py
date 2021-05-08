import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 

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
plt.figure(figsize=(14,5))
plt.suptitle('Sales Predictions')
#subplot1
plt.subplot(1,2,1)
plt.bar(item_top_sales.index, item_top_sales['Item_Outlet_Sales'])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('Item ID')
plt.ylabel('Sales')
plt.title('Top 10 sales by items')

#subplot2
plt.subplot(1,2,2)
plt.bar(outlet_top_sales.index, outlet_top_sales['Item_Outlet_Sales'])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('Outlet ID')
plt.ylabel('Sales')
plt.title('Top 10 sales by outlets')
plt.show()


#backup groups for future use
# item_fat_content = df.groupby(df['Item_Fat_Content'])['Item_Outlet_Sales'].sum()
# item_type = df.groupby(df['Item_Type'])['Item_Outlet_Sales'].sum()
# outlet_year = df.groupby(['Outlet_Establishment_Year'])['Item_Outlet_Sales'].sum()
# outlet_loc = df.groupby(['Outlet_Location_Type'])['Item_Outlet_Sales'].sum()
# outlet_type = df.groupby(['Outlet_Type'])['Item_Outlet_Sales'].sum()