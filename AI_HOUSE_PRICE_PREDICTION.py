import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras

#Load Dataset (Make sure Housing.csv is in the same folder)
data = pd.read_csv("Housing.csv")

#Preprocessing (Keep all rows)
data = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'basement',
             'hotwaterheating', 'airconditioning', 'parking', 'furnishingstatus', 'price']]

#Encode categorical features
label_cols = ['mainroad', 'basement', 'hotwaterheating', 'airconditioning', 'furnishingstatus']
le = LabelEncoder()
for col in label_cols:
    data[col] = le.fit_transform(data[col])

#Define features and label
X = data.drop('price', axis=1)
y = data['price']

#Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Train TensorFlow model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, verbose=0)

#Predict on entire dataset
data['predicted_price'] = model.predict(X_scaled).flatten()

#Compute price-per-sqft and assign ratings
data['price_per_sqft'] = data['predicted_price'] / data['area']
data['rating'] = pd.qcut(data['price_per_sqft'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)  # 5 = best deal

#Make recommendation
median_rating = data['rating'].median()
data['Buy_Recommendation'] = np.where(data['rating'] >= median_rating, 'Buy', 'Don\'t Buy')

#Generate Explanation
def explain(row):
    if row['Buy_Recommendation'] == 'Buy':
        return f"Good deal! ₹{row['predicted_price']:.0f} for {row['area']} sqft. Rating: {row['rating']}/5"
    else:
        return f"Too costly for area: ₹{row['predicted_price']:.0f} for {row['area']} sqft. Rating: {row['rating']}/5"

data['Explanation'] = data.apply(explain, axis=1)

#Show Interactive Bar Chart
fig = px.bar(
    data,
    x=data.index,
    y='rating',
    color='Buy_Recommendation',
    hover_data=['area', 'bedrooms', 'bathrooms', 'stories', 'predicted_price', 'Explanation'],
    labels={'index': 'House #', 'rating': 'Price Rating'},
    title='House Price Ratings (Click for Buy/Don’t Buy Reason)'
)
fig.update_traces(marker_line_width=1.5)
fig.show()

#Display Full Table
print("\nFull House Data with AI Predictions:\n")
table_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'basement',
                 'hotwaterheating', 'airconditioning', 'parking', 'furnishingstatus',
                 'price', 'predicted_price', 'price_per_sqft', 'rating',
                 'Buy_Recommendation', 'Explanation']
print(data[table_columns].to_string(index=True))