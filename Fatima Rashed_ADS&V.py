#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#import data
dataweekly=pd.read_csv("https://raw.githubusercontent.com/FatimaRashed2023/Coursework/main/production_%26_Wastage.csv")


# In[3]:


print(dataweekly)


# In[4]:


dataweekly.shape


# In[5]:


datasales=pd.read_csv("https://raw.githubusercontent.com/FatimaRashed2023/Coursework/main/Cafe%20Sales.csv")


# In[6]:


print(datasales)


# In[7]:


dataweekly.head()


# In[8]:


datasales.head(8)


# In[9]:


dataweekly.tail(2)


# In[10]:


datasales.tail()


# In[11]:


dataweekly.columns


# In[12]:


datasales.columns


# In[13]:


dataweekly.info()


# In[14]:


datasales.info()


# In[15]:


dataweekly.describe()


# In[16]:


datasales.describe()


# In[17]:


#Chceking for any Null values
dataweekly.isna().sum()


# In[18]:


datasales.isna().sum()


# In[19]:


plt.figure(figsize=(15, 10))
plt.bar(dataweekly['Items'], dataweekly['Total Produced'], label='Net Quantity', color='blue')
plt.yticks(dataweekly['Total Produced'][::1])
plt.xlabel('Items')
plt.ylabel('Net Produced')
plt.title('Highest & Lowest items produced')
plt.legend()
plt.xticks(rotation=90, ha='right')
plt.show()


# In[20]:


dataweekly['TotalProducedAndWastage'] = dataweekly['Total Produced'] + dataweekly['Total wastage']
dataweekly_sorted = dataweekly.sort_values(by='TotalProducedAndWastage', ascending=False)

plt.figure(figsize= (15, 5))
plt.barh(dataweekly_sorted['Items'], dataweekly_sorted['Total Produced'], label='Produced', color='limegreen')
plt.barh(dataweekly_sorted['Items'], dataweekly_sorted['Total wastage'], label='wastage', color='maroon')

plt.xlabel('Tota Quantity', fontsize=15) 
plt.ylabel('Item', fontsize=15)
plt.title('Total Produced VS Total Wastage', fontsize=15)
plt.xticks(dataweekly['Total wastage'][::1]) 

plt.xlabel('Total Quantity')
plt.ylabel('List of Items')
plt.title('Total Produced VS Total Wastage')
plt.legend()


# In[21]:


#MOVING TO THE DATA FRAME RELATED TO SALES "datasales"


# In[22]:


# Starting off with calculating the net sales for each category using a Pie chart.
category_net_sales = datasales.groupby('Category')['Net sales'].sum()
plt.figure(figsize=(6, 6))
plt.pie(category_net_sales, labels=category_net_sales.index, autopct='%1.1f%%', startangle=180, colors=plt.cm.Paired.colors)
plt.title('Distribution of Net Sales by Category')
plt.show()


# In[23]:


#Identifying the top 15 selling items from November's sales

top_items = datasales.groupby('Item')['Quantity'].sum().sort_values(ascending=False).head(15)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_items.values, y=top_items.index, palette='viridis')
plt.title('Top 15 Selling Items')
plt.xlabel('Quantity Sold')
plt.show()


# In[24]:


datasales['Date'] = pd.to_datetime(datasales['Date'])

datasales.set_index('Date', inplace=True)
daily_sales = datasales.resample('D')['Net sales'].sum()

plt.figure(figsize=(20, 6))
plt.plot(daily_sales.index, daily_sales.values, linestyle='-', marker='o', color='blue')
plt.title('Daily Net Sales')
plt.xlabel('Days of the month')
plt.ylabel('Net Sales')
plt.show()


# In[25]:


new_data = datasales.query("Category == 'Drinks' and Date == '2023-11-23'")
print(new_data.head())


# In[26]:


new_data.shape


# In[27]:


Drinks_ordered=new_data["Item"].value_counts()
label=Drinks_ordered.index
counts=Drinks_ordered.values
colors=[]


fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Drinks ordered')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='black', width=5)))
fig.show()


# In[28]:


dataweekly.info()


# In[29]:


print(dataweekly)


# In[30]:


item_mapping = {item: idx + 1 for idx, item in enumerate(dataweekly['Items'].unique())}
dataweekly['Items'] = dataweekly['Items'].map(item_mapping)
dataweekly['Items'] = range(1, len(dataweekly) + 1)


# In[31]:


print(dataweekly)


# In[32]:


datasales.info()


# In[33]:


print(datasales)


# In[34]:


datasales["Time"] = datasales["Time"].map({
    "6:01PM_9:00PM": 1,
    "12:00PM_6:00PM": 2,
    "7:00AM_11:59AM": 3
})

datasales["Item"] = datasales["Item"].map({
    "Cappuccino": 1,
    "Standard Takeaway Charges": 2,
    "Bottled Soda": 3,
    "Immunity Juice": 4,
    "Cookies": 5,
    "Cinnamon Roll": 6,
    "Plain Croissants": 7,
    "Kadak Chai": 8,
    "Espresso": 9,
    "Hill Water (Lemon and Mint)": 10,
    "Chicken Fried Rice": 11,
    "SW Chicken Tikka - Grab & Go": 12,
    "Red Blush": 13,
    "Water": 14,
    "Strawberry Classic Shake": 15,
    "Mango Juice 250ml": 16,
    "Strawberry Mojito": 17,
    "Passion Juice 250ml": 18,
    "Mint Pinade": 19,
    "Peanut Butter Smoothie": 20,
    "Kasata (sesame)": 21,
    "Luxury Hot Chocolate": 22,
    "Brownies": 23,
    "Chocolate Muffin": 24,
    "Chicken Stir Fry Noodles": 25,
    "Butter chicken and rice": 26,
    "Iced Tea": 27,
    "Rosho Oat - Choco Brownie": 28,
    "Breakfast (Eggs Your way)": 29,
    "ADD on: Mushrooms": 30,
    "Rosho Oat - Mocha": 31,
    "Chocolate Croissants": 32,
    "Pringles (Salt & Vinegar)": 33,
    "Americano": 34,
    "Swahili bowl": 35,
    "Chai Latte": 36,
    "Salad Chicken tikka - Grab & Go": 37,
    "Kilimanjaro Green Tea": 38,
    "Eggs only": 39,
    "Chicken Biryani Grab&Go": 40,
    "Cafe Latte": 41,
    "Cinnamon Spice": 42,
    "Tropical Mix": 43,
    "Espresso Macchiato": 44,
    "Pringles (Original)": 45,
    "Pringles (Hot & Spicy)": 46,
    "Ginger Mint Tea": 47,
    "Ginger Knock": 48,
    "SW Cheese & Tomato- Grab & Go": 49,
    "Ginger Mint Fusion": 50,
    "Caramel Latte": 51,
    "Vanilla Muffins": 52,
    "Caramel Shake": 53,
    "Rosho Oat - Cinnamon": 54,
    "Herbal Teas": 55,
    "Berry Cool": 56,
    "Turmeric Latte": 57,
    "Iced Coffee": 58,
    "ZO Power": 59,
    "Iced Chai Latte": 60,
    "Vanilla Classic Shake": 61,
    "Cafe Mocha": 62,
    "ZO Shake": 63
})



datasales["Category"] = datasales["Category"].map({
    "Drinks": 1,
    "Takeaways": 2,
    "Pastries": 3,
    "ZO Kitchen - Food": 4,
    "Consumeables Stock": 5
})


print(datasales.head())
print(datasales.tail())


# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

dataweekly_encoded = pd.get_dummies(dataweekly, columns=['Items'], drop_first=True)
dataweekly_encoded = dataweekly_encoded.dropna()  
X = dataweekly_encoded.drop(['Total Produced'], axis=1)
y = dataweekly_encoded['Total wastage']

print(dataweekly.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

forest_model = RandomForestRegressor(n_estimators=50, random_state=42)  
forest_model.fit(X_train, y_train)

# Make predictions
y_pred_forest = forest_model.predict(X_test)
# Evaluate the Random Forest model
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_forest))
print("Random Forest R-squared:", r2_score(y_test, y_pred_forest))


# In[36]:


from sklearn.linear_model import LinearRegression
x = datasales[['Net sales', 'Quantity','Time']]
y = datasales['Category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

a = int(input("Enter the is purchased(1 =500 to 5000, 2 = 5500 to 11500, 3 = more than 11500):"))
b = int(input("The Quantity expected to be Sold(1 = 1, 2 = 2):"))
c = int(input("The Time expected to be Sold(1 = 6:01PM_9:00PM, 2= 12:00PM_6:00PM, 3 = 7:00AM_11:59AM)"))

features = np.array([[a, b, c]])

prediction = model.predict(features)

# Print the predicted time
print("Predict the next sale is from which Category:", prediction)


# In[37]:


from sklearn.linear_model import LinearRegression
x = datasales[['Net sales', 'Quantity','Category']]
y = datasales['Time']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

a = int(input("Enter the is purchased(1 =500 to 5000, 2 = 5500 to 11500, 3 = more than 11500):"))
b = int(input("The Quantity expected to be Sold(1 = 1, 2 = 2):"))
c = int(input("the next sale Category(1 = Drinks, 2= Takeaway, 3 = Pastries, 4 = ZO Kitchen - Food, 5 = Consumable stock)"))

features = np.array([[a, b, c]])

prediction = model.predict(features)

# Print the predicted time
print("Predict The Time to be Sold:", prediction)


# In[38]:


x = dataweekly[['Total Produced', 'Total wastage']]
y = dataweekly['Items']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

a = int(input("The Quantity Produced (1 = 1 to 29, 2 = 30 to 63):"))
b = int(input("Enter the Expected wastage (1 = 1 to 2, 2 = 4 to 5, 3 = 7 to 8, 4 = 11 to 12, 5 = 15 to 17, 6 = 19 to 22):"))

features = np.array([[a, b]])
prediction = model.predict(features)
print("Predicted the item that will be wasted next", prediction)

