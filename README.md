# Smart Loyalty Project

A machine learning-based system for predicting customer loyalty and providing personalized product recommendations using RFM (Recency, Frequency, Monetary) analysis.

## Features

- **Loyalty Prediction**: Predict whether a customer is likely to remain loyal using machine learning
- **RFM Analysis**: Compute customer metrics based on purchase behavior
- **Product Recommendations**: Get personalized product recommendations based on co-occurrence patterns
- **REST API**: Flask backend with JSON endpoints
- **Web Dashboard**: Interactive frontend for testing predictions and recommendations

## Project Structure

```
smart-loyalty-project/
├── backend/               # Flask API backend
│   ├── app.py            # Main Flask application
│   ├── train_loyalty.py  # Model training script
│   └── utils/
│       ├── rfm.py        # RFM feature computation
│       └── basket.py     # Product recommender
├── frontend/             # Web interface
│   ├── index.html        # Home page
│   ├── loyalty.html      # Loyalty prediction page
│   ├── recommendation.html # Product recommendations page
│   ├── css/style.css     # Styling
│   └── js/script.js      # Frontend logic
├── notebooks/            # Data processing scripts
│   ├── cleaning.py       # Data cleaning pipeline
│   ├── eda.ipynb         # Exploratory data analysis
│   └── model.ipynb       # Model training notebook
├── data/                 # Data storage
│   ├── raw/             # Original data
│   └── cleaned/         # Processed data
├── models/              # Trained models and features
├── scripts/             # Utility scripts
│   └── test_api.py      # API tests
└── requirements.txt     # Python dependencies
```

## Run Locally

Follow these steps to set up and run the project:

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Prepare Data

Clean your transaction data:

```powershell
python notebooks/cleaning.py
```

This will:

- Read raw transaction data from `data/raw/sample.csv`
- Clean and validate the data
- Output cleaned data to `data/cleaned/sample_cleaned.csv`

### 3. Compute RFM Features

The RFM features are computed automatically when running the training script, but you can also run it manually:

```powershell
python backend/utils/rfm.py
```

This will output RFM features to `models/rfm_features.csv`.

### 4. Train Loyalty Models

Train the machine learning models:

```powershell
python backend/train_loyalty.py
```

This will:

- Compute RFM features from cleaned data
- Create loyalty labels based on repeat purchase behavior
- Train LogisticRegression and RandomForestClassifier models
- Save the best model to `models/loyalty_model.pkl`

### 5. Start the Flask API

Run the Flask development server:

```powershell
python -m flask --app backend.app run
```

The API will be available at `http://127.0.0.1:5000`

### 6. Access the Web Interface

Open your browser and navigate to:

- **Home**: `http://127.0.0.1:5000/` or open `frontend/index.html`
- **Loyalty Prediction**: `http://127.0.0.1:5000/loyalty.html`
- **Product Recommendations**: `http://127.0.0.1:5000/recommendation.html`

Alternatively, serve the frontend with a static server:

```powershell
# Using Python's built-in server
cd frontend
python -m http.server 8000
# Then open http://localhost:8000
```

## API Endpoints

### Health Check

```bash
curl -X GET http://127.0.0.1:5000/health
```

Response:

```json
{ "status": "ok" }
```

### Predict Loyalty

```bash
curl -X POST http://127.0.0.1:5000/predict-loyalty \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "1002"}'
```

Response:

```json
{
  "customer_id": "1002",
  "loyalty_score": 0.95,
  "loyal": true
}
```

**Parameters:**

- `customer_id` (string, required): The customer ID to predict loyalty for

**Response Fields:**

- `customer_id`: The queried customer ID
- `loyalty_score`: Predicted loyalty score (0-1, higher is more loyal)
- `loyal`: Boolean indicating if loyalty_score >= 0.5

### Get Product Recommendations

```bash
curl -X POST http://127.0.0.1:5000/recommend-products \
  -H "Content-Type: application/json" \
  -d '{"product": "Apple", "top_n": 5}'
```

Response:

```json
{
  "product": "Apple",
  "recommendations": ["Banana", "Orange", "Mango", "Grape", "Kiwi"]
}
```

**Parameters:**

- `product` (string, required): Product name to get recommendations for
- `top_n` (integer, optional): Number of recommendations (default: 5, max: 20)

**Response Fields:**

- `product`: The product you requested recommendations for
- `recommendations`: List of recommended products

## Testing

Run the test suite:

```powershell
pytest scripts/test_api.py -v
```

The tests cover:

- Health check endpoint
- Loyalty prediction endpoint (error handling, field validation)
- Product recommendation endpoint (error handling, response structure)
- CORS headers validation

Tests use fixtures to mock data, so they don't require trained models to run.

## Data Format

### Raw Transaction Data (`data/raw/sample.csv`)

```csv
transaction_id,customer_id,date,product_id,product_name,amount
1,1001,2025-01-01,101,Apple,5.0
2,1002,2025-01-02,102,Banana,3.5
```

### Cleaned Data (`data/cleaned/sample_cleaned.csv`)

```csv
transaction_id,customer_id,date,products,amount
1,1001,2025-01-01,Apple;Banana,8.5
2,1002,2025-01-02,Banana;Orange,12.0
```

### RFM Features (`models/rfm_features.csv`)

```csv
customer_id,recency,frequency,monetary,rfm_score
1001.0,4,1,2.0,0.0375
1002.0,0,2,5.5,1.0000
1003.0,1,1,1.5,0.2250
```

- **Recency**: Days since last purchase (inverted in scoring)
- **Frequency**: Number of purchases
- **Monetary**: Total spending
- **RFM Score**: Weighted composite (R: 0.3, F: 0.4, M: 0.3)

## Configuration

Key files:

- `requirements.txt`: Python package versions
- `backend/app.py`: Flask app configuration (port 5000, CORS enabled)
- `frontend/js/script.js`: API base URL (`http://127.0.0.1:5000`)

To change the API base URL, edit `script.js`:

```javascript
const API_BASE_URL = "http://127.0.0.1:5000"; // Change this
```

## Troubleshooting

### Models not found error

Make sure you've run `python backend/train_loyalty.py` to train models.

### Data not found error

Ensure cleaned data exists at `data/cleaned/sample_cleaned.csv` by running `python notebooks/cleaning.py`.

### Flask import error

Make sure the virtual environment is activated and you're in the project root directory.

### CORS errors in browser

The API includes CORS headers, but ensure you're accessing from the same port or adjust frontend config.

## Notes

- This is a development setup. For production, use a WSGI server like Gunicorn.
- Models are trained on sample data. Replace with your own transaction data for production use.
- The loyalty prediction threshold is 0.5 (customizable in `backend/app.py`).

## License

MIT
