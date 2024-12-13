import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("./dataset/dataset_fix.csv")
print(df.head())

# Preprocessing
df.dropna(axis=0, inplace=True)  # Delete rows with missing values
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

# Label Encoding for categorical features
label_encoder = LabelEncoder()
df['kategori_encoded'] = label_encoder.fit_transform(df['kategori'])
df['provinsi_encoded'] = label_encoder.fit_transform(df['letak_provinsi'])

# Features to be used for clustering
features = ['latitude', 'longitude', 'provinsi_encoded', 'kategori_encoded']
X = df[features]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build a TensorFlow model for clustering or any other purpose
nn_model = Sequential([
    Dense(64, input_dim=X_scaled.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1)  # Adjust as needed for your target output
])

# Compile and train the model
nn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

# Train the model (assuming we're using it for some prediction)
history = nn_model.fit(X_scaled, df['average_rating'], epochs=100, batch_size=64)

# Save the trained model
nn_model.save('./tensorflow_wisata_model.h5')  # Saving the model to h5 file
print("Model saved as 'tensorflow_wisata_model.h5'")

# Function to calculate geodesic distance
def calculate_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geodesic(coords_1, coords_2).kilometers

# Function to get recommended nearby destinations
def recommend_nearby_destinations_with_tensorflow(df, selected_place, num_recommendations=5):
    # Mendapatkan data destinasi yang dipilih
    selected_row = df[df["nama_destinasi"] == selected_place]
    if selected_row.empty:
        return f"Destinasi '{selected_place}' tidak ditemukan dalam dataset."
    
    # Koordinat destinasi yang dipilih
    selected_coords = (selected_row.iloc[0]["latitude"], selected_row.iloc[0]["longitude"])

    # Menghitung prediksi rating untuk semua destinasi
    df['predicted_rating'] = nn_model.predict(X_scaled)

    # Menghitung jarak ke destinasi lain
    df['distance_km'] = df.apply(
        lambda row: calculate_distance(selected_coords[0], selected_coords[1], row["latitude"], row["longitude"]),
        axis=1
    )

    # Mengurutkan berdasarkan jarak terlebih dahulu, kemudian berdasarkan prediksi rating
    recommended_destinations = df.sort_values(by=["distance_km", "predicted_rating"], ascending=[True, False])

    # Menampilkan n rekomendasi teratas
    recommendations = recommended_destinations.head(num_recommendations)

    return recommendations[['nama_destinasi', 'letak_provinsi', 'kategori', 'average_rating', 'predicted_rating', 'distance_km']]

# Menentukan destinasi yang ingin dicari rekomendasinya
selected_place = "Wisata Kampung Marengo Baduy Luar"
recommendations = recommend_nearby_destinations_with_tensorflow(df, selected_place, num_recommendations=3)

# Menampilkan hasil rekomendasi
print("Rekomendasi destinasi terdekat:")
print(recommendations)

# Plotting the results
plt.scatter(df['longitude'], df['latitude'], c=df['predicted_rating'], cmap='viridis')
plt.title('Predicted Rating of Destinations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Predicted Rating')
plt.show()

# Saving the model again in case you need to save it after training
nn_model.save('./tensorflow_wisata_model_with_predictions.h5')  # Save with predictions for future use