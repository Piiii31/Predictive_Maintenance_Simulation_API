from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from datasets import load_dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def create_autoencoder_model():
    data_shape = (16, 48)
    inputs = Input(shape=data_shape)
    encoded = LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False, activation='relu')(inputs)
    decoded = RepeatVector(data_shape[0])(encoded)
    decoded = LSTM(128, return_sequences=True, activation='relu')(decoded)
    decoded = LSTM(data_shape[1], activation='tanh', return_sequences=True)(decoded)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
    return autoencoder

def load_autoencoder_model():
    autoencoder = create_autoencoder_model()
    try:
        autoencoder.load_weights('./Weights/autoencoder_weights.weights.h5', by_name=True, skip_mismatch=True)
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        autoencoder = create_autoencoder_model()
    return autoencoder

autoencoder = load_autoencoder_model()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()  # Parse JSON data
    year = data.get('year')
    quarter = data.get('quarter')

    if not year or not quarter:
        return jsonify({"error": "Year and quarter are required"}), 400

    try:
        # Load and preprocess data
        file_path = f"zip_csv/{year}_{quarter}.zip"
        print(f"Loading dataset from: {file_path}")  # Log the file path

        dataset = load_dataset("backblaze/Drive_Stats", data_files=file_path)
        df = dataset['train'].to_pandas()
        df = df[df.model == 'ST4000DM000']
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(axis='columns', how='all').dropna(axis='rows', how='any')

        # Prepare data
        metadata = df[['serial_number', 'date', 'capacity_bytes', 'model']].copy()
        features = df.drop(columns=['date', 'serial_number', 'model', 'failure', 'capacity_bytes'])

        scaler = StandardScaler()
        x_train = scaler.fit_transform(features)

        # Trim and reshape
        num_samples = x_train.shape[0]
        trimmed_samples = num_samples - (num_samples % 16)
        x_train = x_train[:trimmed_samples, :]
        metadata = metadata.iloc[:trimmed_samples]
        x_train = x_train.reshape(trimmed_samples // 16, 16, x_train.shape[1])

        # Make predictions
        print("Making predictions...")  # Log progress
        y_pred = autoencoder.predict(x_train)
        pred_mse = [mean_squared_error(y_pred[i], x_train[i]) for i in range(len(x_train))]
        threshold = np.percentile(pred_mse, 95)
        pred_res = [0 if mse <= threshold else 1 for mse in pred_mse]

        # Process drives
        rows = []
        for idx, mse in enumerate(pred_mse):
            start_idx = idx * 16
            sequence_metadata = metadata.iloc[start_idx:start_idx + 16]
            rows.append({
                'serial_number': str(sequence_metadata['serial_number'].iloc[0]),
                'model': str(sequence_metadata['model'].iloc[0]),
                'capacity_bytes': int(sequence_metadata['capacity_bytes'].iloc[0]),
                'dates': ",".join(sequence_metadata['date'].dt.strftime('%Y-%m-%d').unique()),
                'status': 'Normal' if pred_res[idx] == 0 else 'Anomalous'
            })

        # Add floor labels to the first 50 drives
        for i in range(50):
            if i < 25:
                rows[i]['floor'] = 1
            else:
                rows[i]['floor'] = 2

        # Return the first 50 drives as JSON
        return jsonify(rows[:50])

    except FileNotFoundError:
        return jsonify({"error": f"File not found: {file_path}"}), 404
    except Exception as e:
        print(f"Error in /analyze: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
