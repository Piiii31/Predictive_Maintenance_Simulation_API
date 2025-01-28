# Predictive Maintenance Simulation API

This repository contains a Flask-based API that simulates predictive maintenance for hard drives using an LSTM autoencoder model. The API loads hard drive data from Backblaze's Drive Stats dataset, preprocesses it, and uses a pre-trained LSTM autoencoder to identify normal and anomalous drives.

## Features

- **Flask API**: The API is built using Flask and supports CORS for cross-origin requests.
- **LSTM Autoencoder**: The model uses an LSTM-based autoencoder to detect anomalies in hard drive data.
- **Backblaze Drive Stats Dataset**: The dataset is loaded based on the specified year and quarter.
- **Preprocessing**: The data is preprocessed to remove missing values and standardize features.
- **Anomaly Detection**: The API identifies normal and anomalous drives based on the reconstruction error from the autoencoder.
- **Output**: The API returns a list of 50 drives with their status (Normal or Anomalous).

## Requirements

- Python 3.x
- Flask
- Flask-CORS
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Datasets (from Hugging Face)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Predictive_Maintenance_Simulation_API.git
   cd Predictive_Maintenance_Simulation_API
   ```



2. Download the pre-trained model weights and place them in the `Weights` directory.

## Usage

1. Run the Flask application:

   ```bash
   python app.py
   ```

   The API will be available at `http://127.0.0.1:5000`.

2. Use the `request.py` script to send a POST request to the API:

   ```bash
   python request.py
   ```

   The script sends a request with the year and quarter (e.g., 2017 Q1) and saves the results to `analyze_results.csv`.

## API Endpoint

### POST `/analyze`

**Request Body:**

```json
{
  "year": "2017",
  "quarter": "Q1"
}
```

**Response:**

- **200 OK**: Returns a JSON list of 50 drives with their status (Normal or Anomalous).
- **400 Bad Request**: If the year or quarter is missing.
- **404 Not Found**: If the dataset file for the specified year and quarter is not found.
- **500 Internal Server Error**: If an error occurs during processing.

**Example Response:**

```json
[
    {
        "serial_number": "Z1E0XXXX",
        "model": "ST4000DM000",
        "capacity_bytes": 4000787030016,
        "dates": "2017-01-01,2017-01-02,...",
        "status": "Normal",
        "floor": 1
    },
    {
        "serial_number": "Z1E0YYYY",
        "model": "ST4000DM000",
        "capacity_bytes": 4000787030016,
        "dates": "2017-01-01,2017-01-02,...",
        "status": "Anomalous",
        "floor": 2
    }
]
```

## Dataset

The dataset used in this project is from Backblaze's Drive Stats, which can be accessed via the Hugging Face Datasets library. The dataset contains detailed information about hard drives, including their serial numbers, models, capacities, and various SMART attributes.

## Model

The LSTM autoencoder model is designed to learn the normal patterns of hard drive data. The model is trained to reconstruct the input data, and the reconstruction error is used to identify anomalies. Drives with high reconstruction errors are flagged as anomalous.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- **Backblaze** for providing the Drive Stats dataset.
- **Hugging Face** for the Datasets library.
- **TensorFlow** for the deep learning framework.

This README provides an overview of the project, its features, and how to use the API. For more details, refer to the code and comments in the repository.
