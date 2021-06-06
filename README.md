# Food Sales Prediction
Predict food sales with machine learning in variety stores of FoodStore.

## Objectives
Analyze the product sales data of FoodStore from 1985 to 2009 and find the factors that affect the sales.

## Factors
- Item Identifier (unique product ID)
- Item Weight
- Item Fat Content (low fat or regular)
- Item Visibility (space used in store)
- Item Type (product category)
- Item MRP
- Outlet Identifier (unique store ID)
- Outlet Establishment Year
- Outlet Size (the size of the store in terms of ground area covered)
- Outlet Location Type (the type of area in which the store is located)
- Outlet Type (grocery store or some sort of supermarket)
- Item Outlet Sales (sales of the product in the particular store)

## Data Clean
### Delete ‘Item_Weight’ column
1463 values are missing in ‘Item_Weight’ column. If delete all the records (rows) where item weight is missing, 17% of the data will be deleted which could impact data analyze. All the missing values in ‘Item_Weight’ are from year 1985. The owner of FoodStore told me that the data was missing because the store didn’t started to record item weight until after 1985. Since the missing data are old and the correlation between item weight and sales are small (0.014, not strongly linear related), I decided to delete the ‘Item_Weight’ column.
### Delete ‘Outlet_Size’ column
2410 values are missing in ‘Outlet_Size’ column. If delete all the records (rows) where outlet size is missing, 28% of the data will be deleted. Not sufficient enough data could cause less accurate data prediction. Those missing values are from Grocery Store and Supermarket Type1. Since each outlet type contains different outlet size, it’s hard to fill the missing values. The histogram showing the sales distribution for ‘Outlet_Size’, ‘Outlet_Location_Type’ and ‘Outlet_Type’ are similar, which means ‘Outlet_Location_Type’ and ‘Outlet_Type’ can represent the outlet features without ‘Outlet_Size’. Also ‘Outlet_Size’ and sales has low correlation (0.13, not strongly linear related)), so I deleted ‘Outlet_Size’ column.
![hist outlet size](https://user-images.githubusercontent.com/82603737/120882587-a426d100-c58d-11eb-8381-bb6f8aee1a76.png)
<img src="https://user-images.githubusercontent.com/82603737/120882589-af79fc80-c58d-11eb-929e-dbff2bd74974.png" width="640" height="480"/>
<img src="https://user-images.githubusercontent.com/82603737/120882596-b4d74700-c58d-11eb-8e72-1ab4f86afbcb.png" width="640" height="480"/>


From correlation heatmap, Item MRP and Outlet Location Type are more related to sales. Outlet Establishment Year and Outlet Size are low correlation with sales.
![corr with annot](https://user-images.githubusercontent.com/82603737/120882886-295eb580-c58f-11eb-877c-9a426d51a3bd.png)
### Delete ‘Outlet_Establishment_Year’ column
Each outlet has an establishment year. Sort sales by outlet. Find out that sales and year are weak correlation (-0.058), which means as the sales increasing or decreasing, the year is randomly displaying, not changing to the same/opposite direction and there’s no pattern. It’s better off to delete the ‘Outlet_Establishment_Year’ column because I don’t want the data analyze model taking recent years as more important than older years. 
![sales by outlet](https://user-images.githubusercontent.com/82603737/120882927-54e1a000-c58f-11eb-8da6-cb624fd850e1.png)

## Data analyze and visualization 
### Item MRP
Item MRP has strong positive correlation (0.59) with sales, meaning that sales increases as the item MRP increases.
![MRP vs Sales](https://user-images.githubusercontent.com/82603737/120882973-88bcc580-c58f-11eb-8bc3-9014574cfee2.png)
### Outlet Location Type
Outlet location type has small positive correlation (0.21) with sales. Although the average sales of 3 locations are close, location tier 3 has more sales in total.
![sales by outlet location type](https://user-images.githubusercontent.com/82603737/120883004-ab4ede80-c58f-11eb-902e-e394833ea8e5.png)
![Loc vs Sales](https://user-images.githubusercontent.com/82603737/120883015-b73aa080-c58f-11eb-8f20-831a477ecf3f.png)
### Outlet Type
Supermarket type 1 has much more sales than other outlet types. 
![sales by outlet type](https://user-images.githubusercontent.com/82603737/120883036-d0dbe800-c58f-11eb-973e-df58cc6a49e0.png)
### Outlet Type and Outlet Location Type
Supermarket Type1 at location Tier 2 has the most sales in all.
![sales by outlet type and location](https://user-images.githubusercontent.com/82603737/120883049-e0f3c780-c58f-11eb-866f-176941bdd9fc.png)
### Item Visibility
Visibility has low negative correlation (-0.11) with sales. The most sales are in lower visibility. As the visibility increasing, the sales is getting lower. The reason could be some items are taken too much capacity than it should be. I also noticed that there are several 0 visibility items with good sales, I assume there are some missing or incorrect data for this feature. 
![visibility vs sales](https://user-images.githubusercontent.com/82603737/120883064-f10ba700-c58f-11eb-96b5-29b304ed37b2.png)

## Modeling
Used train test split on Linear Regression, Random Forest Regression and KNN. Also ran some tests to determine n_estimators for Random Forest Regression and n_neighbors for KNN for best scores. 
### Linear Regression
- score for training data is 0.55.
- score for test data is 0.59.
- RMSE (root mean squared error) is 1093 meaning on average my model predict is off by 1093.

### Random Forest Regression
- n_estimators=80
- score for training data is 0.93.
- score for test data is 0.57.
- RMSE (root mean squared error) is 1121 meaning on average my model predict is off by 1121.

From random forest regression model, item MRP, Outlet Type, Item Visibility are more important features.
![random forest impfeatures](https://user-images.githubusercontent.com/82603737/120883090-18627400-c590-11eb-9b80-62fc2ad26ba3.png)
### KNN
- n_neighbors=13
- score for training data is 0.58.
- score for test data is 0.54.
- RMSE (root mean squared error) is 1149 meaning on average my model predict is off by 1149.
## Conclusion
The final model I choose is Linear Regression because it has the best score for test data – 0.59, meaning 59% of the prediction for unseen data is correct, and lowest RMSE (root mean squared error) – 1093, meaning on average my model is off by 1093. 

From correlation and model analysis, Item MRP, Outlet Location Type, Outlet Type are the main factors affecting sales. 
## Recommendation
- Sell more high price products
- For future stores, supermarket type1 at tier 2 location is the best combination. 


