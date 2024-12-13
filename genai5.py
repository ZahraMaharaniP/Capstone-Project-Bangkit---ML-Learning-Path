import pandas as pd
from geopy.distance import geodesic
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import random
import numpy as np
from datetime import datetime, timedelta

# Load dataset (pastikan path dataset sesuai)
df = pd.read_csv('./dataset/dataset_fix.csv')

# Fungsi untuk menghitung jarak antar destinasi
def calculate_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geodesic(coords_1, coords_2).kilometers

# Membuat model generatif menggunakan TensorFlow
def build_generative_model(vocab_size, embedding_dim=256, lstm_units=512):
    inputs = Input(shape=(None,))
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = Dense(vocab_size, activation='softmax')(x)
    model = Model(inputs, x)
    return model

# Fungsi untuk melatih model generatif
def train_generative_model(model, tokenizer, corpus, epochs=5, batch_size=32):
    # Tokenisasi corpus
    sequences = tokenizer.texts_to_sequences(corpus)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=15, padding='post')

    # Pisahkan inputs dan targets
    inputs = sequences[:, :-1]  # Semua kecuali langkah terakhir
    targets = sequences[:, 1:]  # Semua kecuali langkah pertama
    
    # Pastikan target memiliki dimensi yang sesuai
    targets = tf.expand_dims(targets, axis=-1)  # Tambahkan dimensi untuk kompatibilitas
    
    # Kompilasi model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Melatih model
    model.fit(inputs, targets, epochs=epochs, batch_size=batch_size)

# Fungsi untuk menghasilkan teks menggunakan model generatif
def generate_text(model, tokenizer, prompt, max_length=20):
    input_seq = tokenizer.texts_to_sequences([prompt])[0]  # Tokenize input
    for _ in range(max_length):
        input_array = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=15, padding='post')
        predicted_probs = model.predict(input_array, verbose=0)  # Predict next word probabilities
        predicted_index = np.argmax(predicted_probs[0, -1])  # Look at the last timestep of the sequence
        next_word = tokenizer.index_word.get(predicted_index, '')  # Get the word from index
        if not next_word:  # If no valid word is found, stop generation
            break
        input_seq.append(predicted_index)  # Add word index to sequence
    generated_text = tokenizer.sequences_to_texts([input_seq])[0]
    return generated_text

# Fungsi untuk memilih destinasi secara acak dari dataset berdasarkan tema
def select_destinations_by_theme(theme, region, num_destinations=5):
    filtered_df = df[df['kategori'].str.contains(theme, case=False, na=False)]
    filtered_df = filtered_df[filtered_df['letak_provinsi'] == region]
    selected_destinations = filtered_df.sample(n=num_destinations).drop_duplicates(subset='nama_destinasi')  # Menghapus duplikasi
    return selected_destinations

# Fungsi untuk menghasilkan itinerary lengkap dengan rating, jarak, dan tanggal perencanaan
def generate_full_itinerary(selected_places, theme="alam", region="Jawa Barat", model=None, tokenizer=None, start_date="2024-12-01"):
    prompt = f"Generate an itinerary for {theme} tourism in {region} including: {', '.join(selected_places)}"
    generated_text = generate_text(model, tokenizer, prompt)
    
    # Hapus duplikasi pada selected_places
    selected_places = list(dict.fromkeys(selected_places))  # Menghapus duplikasi dari list
    
    # Inisialisasi itinerary details dengan list kosong
    itinerary_details = []
    visited_destinations = set()  # Set untuk melacak destinasi yang sudah dimasukkan

    current_date = datetime.strptime(start_date, "%Y-%m-%d")  # Mengonversi string start_date ke format datetime
    
    for i in range(len(selected_places) - 1):
        place1 = selected_places[i]
        place2 = selected_places[i + 1]

        # Skip destinasi yang sudah ada di itinerary
        if place1 in visited_destinations:
            continue
        visited_destinations.add(place1)

        if place2 in visited_destinations:
            continue
        visited_destinations.add(place2)
        
        # Get latitude and longitude from dataset for both places
        loc1 = df[df['nama_destinasi'] == place1].iloc[0]
        loc2 = df[df['nama_destinasi'] == place2].iloc[0]
        
        lat1, lon1 = loc1['latitude'], loc1['longitude']
        lat2, lon2 = loc2['latitude'], loc2['longitude']
        
        # Calculate distance
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        
        # Get ratings
        rating1 = loc1['average_rating']
        rating2 = loc2['average_rating']
        
        # Predicted rating (for example, we just calculate an average here, this can be modified)
        predicted_rating1 = (rating1 + rating2) / 2  # Simple predicted rating calculation (modify as needed)
        predicted_rating2 = predicted_rating1  # Assuming we use the same predicted value
        
        # Append the itinerary details with full information
        itinerary_details.append({
            'Nama Destinasi': place1,
            'Letak Provinsi': loc1['letak_provinsi'],
            'Kategori': loc1['kategori'],
            'Average Rating': rating1,
            'Predicted Rating': predicted_rating1,
            'Distance (km)': 0 if i == 0 else distance,  # Distance is zero for the first place
            'Tanggal Perencanaan': current_date.strftime("%Y-%m-%d")  # Format tanggal
        })
        
        itinerary_details.append({
            'Nama Destinasi': place2,
            'Letak Provinsi': loc2['letak_provinsi'],
            'Kategori': loc2['kategori'],
            'Average Rating': rating2,
            'Predicted Rating': predicted_rating2,
            'Distance (km)': distance,
            'Tanggal Perencanaan': (current_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Set tanggal berikutnya
        })
        
        current_date += timedelta(days=1)  # Menambahkan satu hari untuk destinasi berikutnya
    
    # Return the generated text and the detailed itinerary in the new format
    return generated_text, itinerary_details

# Data untuk pelatihan model
corpus = df['nama_destinasi'] + ', ' + df['kategori'] + ', ' + df['letak_provinsi']
corpus = corpus.dropna().tolist()

# Tokenisasi data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)

# Bangun model generatif
vocab_size = len(tokenizer.word_index) + 1
model = build_generative_model(vocab_size)

# Latih model generatif
train_generative_model(model, tokenizer, corpus)

# Contoh penggunaan
theme = "alam"
region = "Jawa Timur"
days = 3
num_destinations = 5
num_versions = 2
start_date = "2024-12-01"  # Misalnya perjalanan dimulai pada 1 Desember 2024

for version in range(1, num_versions + 1):
    selected_places = select_destinations_by_theme(theme, region, num_destinations)['nama_destinasi'].tolist()
    generated_text, itinerary_details = generate_full_itinerary(selected_places, theme, region, model, tokenizer, start_date)

    print(f"Itinerary Version {version}:")
    print(f"Generate a {days}-day itinerary for {theme} tourism in {region}, including the following destinations: {', '.join(selected_places)}.")
    print("Consider travel distance between destinations and highlight natural attractions.\n")
    print(generated_text)
    print("\nDetailed Itinerary:")
    
    # Print the itinerary in the desired format (tabular format)
    itinerary_df = pd.DataFrame(itinerary_details)
    print(itinerary_df.to_string(index=False))
    print("\n")

# Save the model (pastikan file format yang digunakan sesuai dengan standar Keras)
model.save('generative_model_genai5.h5')
