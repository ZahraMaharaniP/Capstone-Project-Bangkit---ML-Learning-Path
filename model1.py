import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf
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
df['popularity_score'] = df['average_rating'] * np.log1p(df['review_total'])

# Prepare features and target
features = ['provinsi_encoded', 'kategori_encoded', 'average_rating', 'review_total']
target = 'popularity_score'
X = df[features]
y = df[target]

# Normalize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Target
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# Build Neural Network Model with Batch Normalization
nn_model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1)
])

# Compile Model
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_schedule = LearningRateScheduler(lambda epoch: 1e-3 * 10**(-epoch / 20))

# Train the Model
history = nn_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,  # Tambahkan jumlah epoch jika overfitting terkendali
    batch_size=64,  # Ukuran batch yang lebih besar untuk stabilitas
    callbacks=[early_stopping, lr_schedule]
)

# Evaluate Model
test_loss = nn_model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss}")

# Save Model
model_path = './model_wisata_popularity_improved.h5'
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