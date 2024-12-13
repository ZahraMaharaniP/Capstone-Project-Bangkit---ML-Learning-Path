import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load dataset
dataset_path = './dataset/dataset_fix.csv'
df = pd.read_csv(dataset_path)

# Preprocessing
df.dropna(inplace=True)
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

# Label Encoding
label_encoder = LabelEncoder()
df['kategori_encoded'] = label_encoder.fit_transform(df['kategori'])
df['provinsi_encoded'] = label_encoder.fit_transform(df['letak_provinsi'])

# Normalize review_total and target variable
scaler = StandardScaler()
df['review_total'] = scaler.fit_transform(df[['review_total']])

# Prepare features and target
features = ['provinsi_encoded', 'kategori_encoded', 'average_rating', 'review_total']
X = df[features]

# Split data into training and testing
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Define the model architecture
provinsi_input = Input(shape=(1,), name='provinsi_encoded')
kategori_input = Input(shape=(1,), name='kategori_encoded')
rating_input = Input(shape=(1,), name='average_rating')
review_input = Input(shape=(1,), name='review_total')

# Embedding layers for categorical features
provinsi_embedding = Embedding(input_dim=df['provinsi_encoded'].nunique(), output_dim=10)(provinsi_input)
kategori_embedding = Embedding(input_dim=df['kategori_encoded'].nunique(), output_dim=10)(kategori_input)

# Flatten the embedding layers
provinsi_flattened = Flatten()(provinsi_embedding)
kategori_flattened = Flatten()(kategori_embedding)

# Concatenate all features
x = Concatenate()([provinsi_flattened, kategori_flattened, rating_input, review_input])

# Dense layers
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
output = Dense(1)(x)

# Define the model
nn_model = Model(inputs=[provinsi_input, kategori_input, rating_input, review_input], outputs=output)

# Compile the model
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

# Train the model
history = nn_model.fit(
    [X_train['provinsi_encoded'], X_train['kategori_encoded'], X_train['average_rating'], X_train['review_total']],
    X_train['average_rating'],  # target
    validation_data=(
        [X_test['provinsi_encoded'], X_test['kategori_encoded'], X_test['average_rating'], X_test['review_total']],
        X_test['average_rating']
    ),
    epochs=100,
    batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)

# Save Model
model_path = './model_wisata_recommendation.h5'
nn_model.save(model_path)
print(f"Model telah disimpan di {model_path}")

# Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# ---- Trying the model with an input destination ----

def try_model_with_input(selected_place):
    # Find the row for the selected destination
    selected_row = df[df['nama_destinasi'] == selected_place]
    
    if selected_row.empty:
        print(f"Destinasi '{selected_place}' tidak ditemukan dalam dataset.")
        return
    
    # Extract features for the selected destination
    provinsi_encoded = selected_row['provinsi_encoded'].values[0]
    kategori_encoded = selected_row['kategori_encoded'].values[0]
    average_rating = selected_row['average_rating'].values[0]
    review_total = selected_row['review_total'].values[0]
    
    # Prepare the input data
    input_data = [
        np.array([provinsi_encoded]),
        np.array([kategori_encoded]),
        np.array([average_rating]),
        np.array([review_total])
    ]
    
    # Predict the popularity score for the selected destination
    predicted_popularity = nn_model.predict(input_data)
    print(f"Predicted Popularity for '{selected_place}': {predicted_popularity[0][0]}")
    
    # Now let's find similar destinations by filtering based on the same category and province
    similar_destinations = df[(df['provinsi_encoded'] == provinsi_encoded) & (df['kategori_encoded'] == kategori_encoded)]
    
    # Sort by rating and review total (for simplicity)
    similar_destinations = similar_destinations.sort_values(by=['average_rating', 'review_total'], ascending=False)
    
    # Show the top 5 similar destinations
    print("\nTop 5 Similar Destinations:")
    print(similar_destinations[['nama_destinasi', 'letak_provinsi', 'kategori', 'average_rating', 'review_total']].head())

# Test with a specific destination
selected_place = "Wisata Mangunan"  # Replace with the destination you want to test
try_model_with_input(selected_place)
