# Software Requirements Specification (SRS)

## Smart Loyalty System - Complete Analysis

**Document Version:** 1.0  
**Date Created:** December 7, 2025  
**Last Updated:** December 7, 2025  
**Project Name:** Smart Loyalty Project  
**Primary Stakeholder:** Academic Institution

---

## 1. Executive Summary

The **Smart Loyalty System** is an intelligent, machine learning-based application designed to predict customer loyalty and provide personalized product recommendations. The system analyzes customer purchase behavior using RFM (Recency, Frequency, Monetary) analysis and deploys machine learning models to generate actionable business insights.

### Key Objectives:

- Predict customer loyalty with high accuracy
- Provide data-driven product recommendations
- Enable data-driven decision-making for retail businesses
- Serve as a research platform for academic institutions

---

## 2. Project Overview

### 2.1 Purpose

The Smart Loyalty System enables organizations to:

1. Identify and retain loyal customers
2. Predict customer churn risk
3. Generate personalized product recommendations
4. Optimize marketing strategies based on customer segments

### 2.2 Scope

**Included:**

- Customer loyalty prediction using machine learning
- Product recommendation engine
- REST API for programmatic access
- Web dashboard for visualization and testing
- RFM feature computation and analysis
- Model training and evaluation

**Excluded:**

- Email marketing automation
- Payment processing integration
- Customer CRM system
- Real-time data streaming

### 2.3 Stakeholders

| Stakeholder          | Role         | Interests                                     |
| -------------------- | ------------ | --------------------------------------------- |
| Academic Institution | Primary User | Research, validation, educational use         |
| Data Analysts        | System Users | Data insights, model performance              |
| Retail Managers      | End Users    | Customer loyalty predictions, recommendations |
| Developers           | Maintainers  | Code quality, system reliability              |

---

## 3. Current Project Structure

### 3.1 Directory Hierarchy

```
smart-loyalty-project/
├── backend/                          # Flask REST API backend
│   ├── app.py                       # Main Flask application (150+ lines)
│   ├── app_simple.py                # Simplified Flask variant
│   ├── train_loyalty.py             # Model training script (223 lines)
│   ├── temp.py                      # Temporary/development file
│   ├── __init__.py                  # Python package initialization
│   ├── routes/                      # API route handlers
│   │   ├── api.py                   # API endpoints
│   │   └── __init__.py
│   └── utils/                       # Utility modules
│       ├── rfm.py                   # RFM computation (124 lines)
│       ├── basket.py                # Product recommender (83 lines)
│       └── __init__.py
├── frontend/                         # Web user interface
│   ├── index.html                   # Landing page (60+ lines)
│   ├── loyalty.html                 # Loyalty prediction UI
│   ├── recommendation.html          # Recommendation UI
│   ├── css/
│   │   └── style.css                # Stylesheet
│   └── js/
│       └── script.js                # Frontend logic
├── notebooks/                        # Data processing & analysis
│   ├── cleaning.py                  # Data cleaning pipeline
│   ├── eda.ipynb                    # Exploratory Data Analysis
│   └── model.ipynb                  # Model training notebook
├── data/                             # Data storage
│   ├── raw/                         # Original transaction data
│   └── cleaned/                     # Processed transaction data
│       └── sample_cleaned.csv       # Cleaned sample data
├── models/                           # Trained models & features
│   ├── loyalty_model.pkl            # Trained ML model
│   ├── pipeline.pkl                 # Preprocessing pipeline (optional)
│   ├── rfm_features.csv             # Computed RFM features
│   └── rf_feature_importances.csv   # Feature importance analysis
├── scripts/                          # Utility scripts
│   └── test_api.py                  # API endpoint testing
├── report/                           # Documentation & analysis reports
│   └── MODEL-EVALUATION-GUIDE.md
├── src/                              # New modular source structure (optional)
├── README.md                         # Project documentation (277 lines)
├── requirements.txt                 # Python dependencies
└── .venv/                           # Python virtual environment

```

### 3.2 File Statistics

| Component     | File Count           | Total Lines | Purpose                         |
| ------------- | -------------------- | ----------- | ------------------------------- |
| Backend       | 3 core files + utils | 500+        | API endpoints, model management |
| Frontend      | 3 HTML + CSS + JS    | 200+        | User interface                  |
| Models        | 4 files              | N/A         | Serialized models & features    |
| Data          | 2 folders            | N/A         | Raw and processed data          |
| Documentation | 2 files              | 400+        | Project info and guides         |

---

## 4. Core Features & Functionality

### 4.1 Loyalty Prediction Module

**Purpose:** Predict whether a customer will remain loyal based on purchase history

**Inputs:**

- Customer ID (string or numeric)
- RFM Features:
  - **Recency:** Days since last purchase
  - **Frequency:** Number of purchases
  - **Monetary:** Total spending amount
  - **RFM Score:** Weighted combination (R: 30%, F: 40%, M: 30%)

**Outputs:**

- Loyalty Score (0.0 - 1.0 probability)
- Binary Classification (Loyal/Not Loyal at 0.5 threshold)

**Algorithm:**

- Primary: Logistic Regression or Random Forest Classifier
- Preprocessing: StandardScaler normalization
- Model Path: `models/loyalty_model.pkl`

**API Endpoint:**

```
POST /predict-loyalty
Content-Type: application/json

{
  "customer_id": "12345"
}

Response:
{
  "customer_id": "12345",
  "loyalty_score": 0.78,
  "loyal": true
}
```

### 4.2 Product Recommendation Module

**Purpose:** Generate personalized product recommendations based on co-occurrence patterns

**Strategy:**

1. **Multi-product transactions:** Find products purchased together
2. **Customer history:** Find products bought by customers who bought the target product

**Outputs:**

- List of recommended products (top N, default: 5)
- Ranked by co-occurrence frequency

**API Endpoint:**

```
POST /recommend-products
Content-Type: application/json

{
  "product": "Product Name",
  "top_n": 5
}

Response:
{
  "product": "Product Name",
  "recommendations": [
    "Product A",
    "Product B",
    "Product C",
    "Product D",
    "Product E"
  ]
}
```

### 4.3 RFM Feature Engineering

**Purpose:** Compute Recency, Frequency, and Monetary metrics for each customer

**Input Data:**

- Cleaned transaction CSV with columns: `transaction_id`, `customer_id`, `date`, `products`, `amount`

**Processing:**

1. Parse transaction dates
2. Set reference date (default: max date in dataset)
3. For each customer:
   - **Recency:** Days since reference date to last purchase
   - **Frequency:** Total number of transactions
   - **Monetary:** Sum of transaction amounts
4. Min-max normalize each metric (0-1 range)
5. Compute weighted RFM Score: `R*0.3 + F*0.4 + M*0.3`

**Output:**

- CSV with columns: `customer_id`, `recency`, `frequency`, `monetary`, `rfm_score`
- File: `models/rfm_features.csv`

### 4.4 Loyalty Labels Generation

**Purpose:** Create training labels for supervised learning

**Methodology:**

1. Set reference date (cutoff for feature computation)
2. Identify customers with transactions before/on reference date
3. Define prediction window (typically 60 days after reference date)
4. Label: 1 if customer made purchase in window, 0 otherwise
5. This follows standard ML practice: train on historical data, predict future behavior

**Logic:**

- Reference Date: Configurable, default to `max_date - 60 days`
- Label Window: Configurable, default 60 days
- Output: `customer_id`, `label` (0/1), `reference_date`

---

## 5. Technical Architecture

### 5.1 Backend Stack

**Framework:** Flask

- Lightweight Python web framework
- JSON request/response handling
- CORS support for cross-origin requests
- Development server included

**ML Libraries:**

- `scikit-learn`: Machine learning models (LogisticRegression, RandomForestClassifier)
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `joblib`: Model serialization

**Dependencies:**

```
flask              # Web framework
scikit-learn       # ML algorithms
pandas             # Data processing
numpy              # Numerical computing
mlxtend            # Extended ML utilities
joblib             # Model persistence
matplotlib         # Data visualization
```

### 5.2 Frontend Stack

**Technologies:**

- HTML5: Semantic markup
- CSS3: Responsive styling
- Vanilla JavaScript: Client-side logic
- No frontend framework (lightweight, self-contained)

**Pages:**

1. **index.html** (60+ lines): Landing page with feature overview
2. **loyalty.html**: Interactive loyalty prediction interface
3. **recommendation.html**: Product recommendation interface

**Features:**

- Form validation
- API integration via fetch()
- Real-time result display
- Responsive design

### 5.3 Data Pipeline

```
Data Sources
    ↓
data/raw/ [Raw Transaction Data]
    ↓
notebooks/cleaning.py [Data Cleaning & Validation]
    ↓
data/cleaned/sample_cleaned.csv [Cleaned Transactions]
    ↓
─────────────────────────────────
│ RFM Computation (backend/utils/rfm.py)
│ └─→ models/rfm_features.csv
└─→ Loyalty Label Generation (backend/train_loyalty.py)
    │
    └─→ Model Training
        ├─ Preprocessing (StandardScaler)
        ├─ Train/Test Split (default: 80/20)
        ├─ Model Fitting (LogisticRegression + RandomForestClassifier)
        └─→ models/loyalty_model.pkl [Best Model]
```

### 5.4 API Architecture

**Base URL:** `http://127.0.0.1:5000`

**Core Endpoints:**

| Method | Endpoint              | Purpose                     | Input              | Output                   |
| ------ | --------------------- | --------------------------- | ------------------ | ------------------------ |
| GET    | `/`                   | Serve landing page          | -                  | HTML                     |
| GET    | `/health`             | System health check         | -                  | `{status: ok}`           |
| POST   | `/predict-loyalty`    | Predict customer loyalty    | `{customer_id}`    | `{loyalty_score, loyal}` |
| POST   | `/recommend-products` | Get product recommendations | `{product, top_n}` | `{recommendations}`      |

**CORS Configuration:**

- Allows: All origins (`*`)
- Methods: GET, POST, OPTIONS
- Headers: Content-Type, Authorization

---

## 6. Data Schema

### 6.1 Input Data Format

**Raw Transaction Data** (`data/raw/sample.csv`)

| Column         | Type         | Description                              | Example                              |
| -------------- | ------------ | ---------------------------------------- | ------------------------------------ |
| transaction_id | String/Int   | Unique transaction identifier            | TX001, TX002                         |
| customer_id    | String/Float | Customer unique identifier               | C123, C456                           |
| date           | Date         | Transaction date                         | 2024-01-15                           |
| products       | String       | Product name or semicolon-separated list | "Product A" or "Product A;Product B" |
| amount         | Float        | Transaction amount                       | 49.99, 150.00                        |

### 6.2 Cleaned Data Format

**Cleaned Transaction Data** (`data/cleaned/sample_cleaned.csv`)

Same as raw data with:

- Invalid dates removed/parsed
- Null/empty values handled
- Data type consistency enforced
- Date formatted as ISO 8601

### 6.3 RFM Features

**RFM Features Table** (`models/rfm_features.csv`)

| Column      | Type         | Range | Description                                    |
| ----------- | ------------ | ----- | ---------------------------------------------- |
| customer_id | String/Float | -     | Customer identifier                            |
| recency     | Float        | 0-1   | Normalized days since last purchase (inverted) |
| frequency   | Float        | 0-1   | Normalized purchase count                      |
| monetary    | Float        | 0-1   | Normalized total spending                      |
| rfm_score   | Float        | 0-1   | Weighted combination: R*0.3 + F*0.4 + M\*0.3   |

### 6.4 Model Input/Output

**Prediction Request**

```json
{
  "customer_id": "12345"
}
```

**Prediction Response**

```json
{
  "customer_id": "12345",
  "loyalty_score": 0.78,
  "loyal": true
}
```

**Recommendation Request**

```json
{
  "product": "Laptop",
  "top_n": 5
}
```

**Recommendation Response**

```json
{
  "product": "Laptop",
  "recommendations": [
    "Mouse",
    "Keyboard",
    "Monitor",
    "USB Cable",
    "Laptop Stand"
  ]
}
```

---

## 7. Machine Learning Models

### 7.1 Model Training Pipeline

**Script:** `backend/train_loyalty.py`

**Process:**

1. **Load RFM Features:** `models/rfm_features.csv`
2. **Generate Loyalty Labels:**
   - Reference date: `max_date - 60 days` (default)
   - Prediction window: 60 days forward
   - Label: Binary (loyal/not loyal)
3. **Merge Labels with RFM:** Create training dataset
4. **Feature Selection:** Columns: `['recency', 'frequency', 'monetary', 'rfm_score']`
5. **Preprocessing Pipeline:**
   - StandardScaler: Mean=0, Std=1 normalization
6. **Train/Test Split:** 80% training, 20% testing
7. **Model Training:**
   - Model 1: LogisticRegression
   - Model 2: RandomForestClassifier
8. **Model Evaluation:**
   - Metrics: Accuracy, Precision, Recall, ROC-AUC
   - Select best performing model
9. **Save Model:** `models/loyalty_model.pkl` (joblib format)

### 7.2 Model Specifications

**Algorithm 1: Logistic Regression**

- Type: Linear classification
- Pros: Fast, interpretable, low computational cost
- Cons: Assumes linear separability
- Best for: Quick predictions, simple patterns

**Algorithm 2: Random Forest Classifier**

- Type: Ensemble decision trees
- Pros: Handles non-linear relationships, feature importance ranking
- Cons: Slower prediction, memory intensive for large datasets
- Best for: Complex patterns, feature analysis

**Selected Model:** Whichever achieves higher ROC-AUC score

### 7.3 Feature Importance

**Output:** `models/rf_feature_importances.csv`

Identifies which RFM metrics are most predictive:

- If Random Forest selected: Provides feature importance scores
- Helps business understand which customer behaviors predict loyalty

---

## 8. Data Processing Pipeline

### 8.1 Data Cleaning

**Script:** `notebooks/cleaning.py`

**Operations:**

1. Read raw CSV file
2. Validate column presence
3. Parse and standardize dates
4. Remove null/empty transaction rows
5. Validate numeric fields (amount, customer_id)
6. Remove duplicates (if any)
7. Write cleaned data to `data/cleaned/`

**Input:** `data/raw/sample.csv`  
**Output:** `data/cleaned/sample_cleaned.csv`

### 8.2 Exploratory Data Analysis

**Notebook:** `notebooks/eda.ipynb`

**Analysis:**

- Data shape and types
- Missing value analysis
- Customer distribution
- Purchase amount statistics
- Temporal patterns
- Top products analysis
- Visualizations (histograms, box plots, time series)

### 8.3 Model Development Notebook

**Notebook:** `notebooks/model.ipynb`

**Contains:**

- Data loading and exploration
- RFM feature computation
- Label generation
- Model training
- Performance evaluation
- Result visualization

---

## 9. Functional Requirements

### 9.1 Loyalty Prediction FR

| ID      | Requirement                  | Priority | Details                                   |
| ------- | ---------------------------- | -------- | ----------------------------------------- |
| FR-LP-1 | Accept customer ID input     | HIGH     | Support both string and numeric formats   |
| FR-LP-2 | Retrieve RFM features        | HIGH     | Load from `models/rfm_features.csv`       |
| FR-LP-3 | Compute loyalty probability  | HIGH     | Use trained model for prediction          |
| FR-LP-4 | Return loyalty score (0-1)   | HIGH     | Float value indicating loyalty likelihood |
| FR-LP-5 | Return binary classification | HIGH     | Loyal (score ≥ 0.5) or Not Loyal (< 0.5)  |
| FR-LP-6 | Handle missing customers     | MEDIUM   | Return error message with available IDs   |
| FR-LP-7 | Validate input format        | MEDIUM   | Ensure JSON request format                |

### 9.2 Product Recommendation FR

| ID      | Requirement                    | Priority | Details                                       |
| ------- | ------------------------------ | -------- | --------------------------------------------- |
| FR-PR-1 | Accept product name input      | HIGH     | Case-insensitive matching                     |
| FR-PR-2 | Accept top_n parameter         | MEDIUM   | Default 5, customizable                       |
| FR-PR-3 | Compute co-occurrence patterns | HIGH     | Support 2 strategies (transaction & customer) |
| FR-PR-4 | Return ranked recommendations  | HIGH     | Sorted by co-occurrence frequency             |
| FR-PR-5 | Handle missing products        | MEDIUM   | Return empty list gracefully                  |
| FR-PR-6 | Validate input format          | MEDIUM   | Ensure JSON request format                    |

### 9.3 RFM Computation FR

| ID       | Requirement             | Priority | Details                                      |
| -------- | ----------------------- | -------- | -------------------------------------------- |
| FR-RFM-1 | Parse transaction dates | HIGH     | Handle multiple date formats                 |
| FR-RFM-2 | Calculate recency       | HIGH     | Days since last purchase from reference date |
| FR-RFM-3 | Calculate frequency     | HIGH     | Total purchase count per customer            |
| FR-RFM-4 | Calculate monetary      | HIGH     | Total spending per customer                  |
| FR-RFM-5 | Normalize metrics       | HIGH     | Min-max scaling to 0-1 range                 |
| FR-RFM-6 | Compute RFM score       | HIGH     | Weighted formula: R*0.3 + F*0.4 + M\*0.3     |
| FR-RFM-7 | Export features         | HIGH     | Save to CSV format                           |

### 9.4 Model Training FR

| ID      | Requirement             | Priority | Details                                      |
| ------- | ----------------------- | -------- | -------------------------------------------- |
| FR-MT-1 | Generate loyalty labels | HIGH     | Based on reference date and window           |
| FR-MT-2 | Handle date edge cases  | MEDIUM   | Missing dates, small datasets                |
| FR-MT-3 | Train multiple models   | HIGH     | LogisticRegression & RandomForest            |
| FR-MT-4 | Evaluate performance    | HIGH     | Compute accuracy, precision, recall, ROC-AUC |
| FR-MT-5 | Select best model       | HIGH     | Choose based on ROC-AUC score                |
| FR-MT-6 | Serialize model         | HIGH     | Save to joblib format (.pkl)                 |
| FR-MT-7 | Print metrics           | MEDIUM   | Display training results to console          |

### 9.5 Web UI FR

| ID      | Requirement              | Priority | Details                          |
| ------- | ------------------------ | -------- | -------------------------------- |
| FR-UI-1 | Display landing page     | HIGH     | Show features and navigation     |
| FR-UI-2 | Accept customer ID input | HIGH     | Loyalty prediction form          |
| FR-UI-3 | Accept product input     | HIGH     | Recommendation form              |
| FR-UI-4 | Display results          | HIGH     | Show predictions/recommendations |
| FR-UI-5 | Show error messages      | MEDIUM   | Inform users of invalid input    |
| FR-UI-6 | Responsive design        | MEDIUM   | Work on mobile and desktop       |

---

## 10. Non-Functional Requirements

### 10.1 Performance Requirements

| Requirement               | Target            | Priority |
| ------------------------- | ----------------- | -------- |
| API response time         | < 500ms           | HIGH     |
| Model prediction latency  | < 100ms           | HIGH     |
| Recommendation generation | < 200ms           | MEDIUM   |
| Frontend page load        | < 2 seconds       | MEDIUM   |
| Support concurrent users  | 100+ simultaneous | MEDIUM   |

### 10.2 Reliability Requirements

| Requirement             | Specification                     | Priority |
| ----------------------- | --------------------------------- | -------- |
| Model availability      | 99%+ uptime                       | HIGH     |
| Graceful error handling | All endpoints return JSON errors  | HIGH     |
| Data validation         | Input validation on all endpoints | HIGH     |
| Fallback behavior       | Return sensible defaults on error | MEDIUM   |

### 10.3 Scalability Requirements

| Requirement         | Specification            | Priority |
| ------------------- | ------------------------ | -------- |
| Support data growth | Handle 100K+ customers   | MEDIUM   |
| RFM computation     | Process 1M+ transactions | MEDIUM   |
| Model retraining    | Support batch updates    | LOW      |

### 10.4 Security Requirements

| Requirement      | Specification              | Priority |
| ---------------- | -------------------------- | -------- |
| CORS policy      | Allow specified origins    | MEDIUM   |
| Input validation | Sanitize all user inputs   | HIGH     |
| Error messages   | No sensitive data exposure | HIGH     |
| Data protection  | No hardcoded credentials   | MEDIUM   |

### 10.5 Maintainability Requirements

| Requirement              | Specification                  | Priority |
| ------------------------ | ------------------------------ | -------- |
| Code documentation       | Module docstrings and comments | MEDIUM   |
| Configuration management | Centralized settings           | MEDIUM   |
| Testing framework        | Unit and integration tests     | LOW      |
| Logging                  | Debug and error logging        | MEDIUM   |

---

## 11. Current Implementation Status

### 11.1 Completed Features ✅

- ✅ Flask REST API with core endpoints
- ✅ Loyalty prediction model training
- ✅ RFM feature computation
- ✅ Product recommendation engine
- ✅ Web dashboard with HTML/CSS/JS
- ✅ Model serialization (joblib)
- ✅ Data cleaning pipeline
- ✅ CORS support
- ✅ Error handling for API endpoints

### 11.2 Partially Implemented Features ⚠️

- ⚠️ Comprehensive error logging (basic exists)
- ⚠️ Unit testing (test_api.py exists but may be incomplete)
- ⚠️ Data validation (basic exists)

### 11.3 Planned/Not Implemented Features ❌

- ❌ Authentication and authorization
- ❌ Database integration (currently file-based)
- ❌ Real-time model updates
- ❌ Advanced monitoring and alerting
- ❌ API documentation (Swagger/OpenAPI)
- ❌ Rate limiting
- ❌ Caching layer

---

## 12. Usage Guide

### 12.1 Setup Instructions

```bash
# 1. Navigate to project directory
cd smart-loyalty-project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### 12.2 Data Preparation

```bash
# 1. Place raw transaction data
# File: data/raw/sample.csv
# Columns: transaction_id, customer_id, date, products, amount

# 2. Run data cleaning
python notebooks/cleaning.py
# Output: data/cleaned/sample_cleaned.csv
```

### 12.3 Model Training

```bash
# 1. Compute RFM features
python backend/utils/rfm.py
# Output: models/rfm_features.csv

# 2. Train loyalty models
python backend/train_loyalty.py
# Output: models/loyalty_model.pkl
# Prints: Accuracy, Precision, Recall, ROC-AUC scores
```

### 12.4 Running the Application

```bash
# Start Flask development server
python -m flask --app backend.app run

# Server will start at: http://127.0.0.1:5000
# Access web dashboard at: http://127.0.0.1:5000
```

### 12.5 Testing API Endpoints

```bash
# Run API tests
python scripts/test_api.py

# Or test manually using curl:
curl -X POST http://127.0.0.1:5000/predict-loyalty \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "12345"}'

curl -X POST http://127.0.0.1:5000/recommend-products \
  -H "Content-Type: application/json" \
  -d '{"product": "Laptop", "top_n": 5}'
```

---

## 13. Known Issues & Limitations

### 13.1 Current Issues

| Issue                             | Severity | Description                                    | Workaround                         |
| --------------------------------- | -------- | ---------------------------------------------- | ---------------------------------- |
| Date parsing errors               | MEDIUM   | Some date formats may not parse correctly      | Use ISO 8601 format (YYYY-MM-DD)   |
| Missing customer handling         | LOW      | Returns error with available IDs (may be long) | Improve error message formatting   |
| RFM normalization edge case       | LOW      | Constant values in RFM return 0                | Handle in downstream code          |
| Model not serialized on first run | MEDIUM   | Must run training script first                 | Implement auto-training on startup |

### 13.2 Limitations

| Limitation                   | Impact                  | Potential Solution                      |
| ---------------------------- | ----------------------- | --------------------------------------- |
| File-based storage           | Limited scalability     | Integrate database (PostgreSQL/MongoDB) |
| Single model per prediction  | No A/B testing          | Implement model versioning              |
| Static RFM weights           | Inflexible scoring      | Make weights configurable               |
| No caching                   | Performance degradation | Implement Redis caching                 |
| No authentication            | Security risk           | Add API key/OAuth support               |
| Batch prediction inefficient | Slow for bulk requests  | Add batch endpoint                      |

---

## 14. Code Quality Metrics

### 14.1 File Statistics

| Module                   | Lines | Functions | Classes | Complexity |
| ------------------------ | ----- | --------- | ------- | ---------- |
| backend/app.py           | 150+  | 8+        | 1 (app) | Medium     |
| backend/train_loyalty.py | 223   | 4+        | 0       | Medium     |
| backend/utils/rfm.py     | 124   | 3         | 0       | Low        |
| backend/utils/basket.py  | 83    | 2         | 0       | Low        |
| frontend/index.html      | 60+   | N/A       | N/A     | Low        |

### 14.2 Code Quality Observations

**Strengths:**

- Clear function documentation with docstrings
- Type hints in function signatures
- Error handling in API endpoints
- Modular organization (utils, routes)
- Comments explaining complex logic

**Areas for Improvement:**

- Add comprehensive unit tests
- Improve logging (currently minimal)
- Add input validation layer
- Extract magic numbers to constants
- Add configuration management
- Implement proper exception hierarchy

---

## 15. Deployment Considerations

### 15.1 Development vs Production

**Development:**

```bash
python -m flask --app backend.app run
# Runs on 127.0.0.1:5000
# Debug mode: ON
# Hot reload: ENABLED
```

**Production:**

- Use production WSGI server (Gunicorn, uWSGI)
- Set DEBUG=False
- Use environment variables for configuration
- Implement proper logging
- Use database instead of CSV files
- Implement caching layer
- Add API rate limiting

### 15.2 Environment Variables

Recommended configuration:

```bash
FLASK_ENV=production
DEBUG=False
MODEL_PATH=/path/to/models/
DATA_PATH=/path/to/data/
LOG_LEVEL=INFO
```

### 15.3 Dependencies Management

**Current requirements.txt:**

```
flask
scikit-learn
pandas
numpy
mlxtend
joblib
matplotlib
```

**Recommended production additions:**

```
gunicorn              # Production WSGI server
python-dotenv         # Environment variable management
pytest                # Testing framework
pytest-cov            # Coverage reporting
black                 # Code formatting
flake8                # Linting
mypy                  # Type checking
```

---

## 16. Testing Strategy

### 16.1 Test Coverage

| Component             | Type        | Status     | Coverage    |
| --------------------- | ----------- | ---------- | ----------- |
| API Endpoints         | Integration | ⚠️ Partial | ~60%        |
| RFM Computation       | Unit        | ❌ None    | 0%          |
| Model Training        | Integration | ❌ None    | 0%          |
| Recommendation Engine | Unit        | ❌ None    | 0%          |
| Frontend              | Manual      | ⚠️ Basic   | Manual only |

### 16.2 Recommended Tests

**Unit Tests:**

- RFM feature computation accuracy
- Normalization correctness
- Recommendation algorithm correctness
- Error handling

**Integration Tests:**

- API endpoint functionality
- End-to-end prediction workflow
- Data pipeline integration
- Model loading and inference

**Performance Tests:**

- Prediction latency
- Batch processing speed
- Memory usage under load

---

## 17. Documentation Structure

| Document    | Location                           | Purpose                                     |
| ----------- | ---------------------------------- | ------------------------------------------- |
| README      | `README.md`                        | Project overview and quick start            |
| SRS         | `SRS-DOCUMENT.md`                  | Requirements and specifications (this file) |
| Model Guide | `report/MODEL-EVALUATION-GUIDE.md` | Model evaluation details                    |
| API Docs    | `backend/app.py` (docstrings)      | API endpoint documentation                  |
| Data Schema | `README.md`                        | Data format specification                   |

---

## 18. Success Criteria

### 18.1 Functional Success Criteria

- ✅ Loyalty prediction accuracy > 75%
- ✅ API response time < 500ms
- ✅ All endpoints return proper JSON format
- ✅ Error handling covers all edge cases
- ✅ Web dashboard displays results correctly

### 18.2 Non-Functional Success Criteria

- ✅ System handles 100+ concurrent users
- ✅ 99%+ API availability
- ✅ Code maintainability score > 80
- ✅ All dependencies documented
- ✅ Security best practices implemented

### 18.3 Academic Success Criteria

- ✅ Comprehensive documentation
- ✅ Reproducible results
- ✅ Clear methodology explanation
- ✅ Performance metrics reporting
- ✅ Future improvement suggestions

---

## 19. Future Enhancements

### 19.1 Short-term (1-3 months)

1. **Testing Framework**

   - Add pytest for unit tests
   - Achieve 80%+ code coverage
   - Implement CI/CD pipeline

2. **Configuration Management**

   - Extract hardcoded values to config
   - Support multiple environments
   - Add environment variable support

3. **Logging and Monitoring**
   - Implement comprehensive logging
   - Add application performance monitoring
   - Create monitoring dashboard

### 19.2 Medium-term (3-6 months)

1. **Database Integration**

   - Replace CSV storage with database
   - Implement data persistence layer
   - Add schema migrations

2. **API Improvements**

   - Add OpenAPI/Swagger documentation
   - Implement request/response validation
   - Add batch prediction endpoint
   - Implement caching layer

3. **Model Enhancements**
   - Implement model versioning
   - Add A/B testing capability
   - Support model retraining pipeline
   - Add feature store

### 19.3 Long-term (6+ months)

1. **Advanced ML Features**

   - Add deep learning models
   - Implement ensemble methods
   - Support real-time prediction updates
   - Add anomaly detection

2. **Enterprise Features**

   - Add authentication/authorization
   - Implement role-based access control
   - Add audit logging
   - Implement data encryption

3. **Scalability**
   - Containerize with Docker
   - Implement Kubernetes deployment
   - Add message queue (RabbitMQ/Kafka)
   - Implement microservices architecture

---

## 20. Glossary & Terminology

| Term                  | Definition                                                                |
| --------------------- | ------------------------------------------------------------------------- |
| **Recency**           | Number of days since a customer's most recent purchase                    |
| **Frequency**         | Number of times a customer has made a purchase                            |
| **Monetary**          | Total amount spent by a customer                                          |
| **RFM Score**         | Weighted combination of Recency, Frequency, and Monetary metrics          |
| **Loyalty**           | Binary classification indicating if a customer will make future purchases |
| **Co-occurrence**     | Products frequently purchased together by customers                       |
| **Prediction Window** | Time period used for generating loyalty labels                            |
| **Reference Date**    | Cutoff date for computing features and generating labels                  |
| **Precision**         | Proportion of predicted loyal customers who are actually loyal            |
| **Recall**            | Proportion of actual loyal customers correctly identified                 |
| **ROC-AUC**           | Area Under the Receiver Operating Characteristic Curve                    |
| **CORS**              | Cross-Origin Resource Sharing - allows cross-domain API requests          |
| **JSON**              | JavaScript Object Notation - lightweight data format                      |
| **Joblib**            | Python library for serializing and deserializing Python objects           |
| **API Endpoint**      | Specific URL path that handles a particular request                       |

---

## 21. References & Resources

### 21.1 Related Documents

- README.md - Project overview
- MODEL-EVALUATION-GUIDE.md - Model evaluation details
- requirements.txt - Dependency specifications

### 21.2 External Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [RFM Analysis Guide](<https://en.wikipedia.org/wiki/RFM_(customer_value)>)

### 21.3 Best Practices

- Machine Learning Operations (MLOps)
- API Design Best Practices
- Data Science Workflow Documentation
- Software Development Lifecycle Standards

---

## Appendix A: API Reference Summary

### Health Check

```
GET /health
Response: {"status": "ok"}
```

### Predict Loyalty

```
POST /predict-loyalty
Request: {"customer_id": "string|number"}
Response: {"customer_id": "...", "loyalty_score": 0-1, "loyal": boolean}
Errors: 400 (missing customer), 500 (model unavailable)
```

### Recommend Products

```
POST /recommend-products
Request: {"product": "string", "top_n": "integer"}
Response: {"product": "...", "recommendations": ["list", "of", "products"]}
Errors: 400 (missing product), 500 (recommendation error)
```

---

## Appendix B: Data Format Examples

### Sample Raw Transaction Data

```csv
transaction_id,customer_id,date,products,amount
TX001,C001,2024-01-10,Laptop,999.99
TX002,C001,2024-01-15,Mouse;Keyboard,49.99
TX003,C002,2024-01-12,Monitor,299.99
TX004,C002,2024-01-20,USB Cable,9.99
```

### Sample RFM Features

```csv
customer_id,recency,frequency,monetary,rfm_score
C001,0.85,0.90,0.88,0.88
C002,0.75,0.60,0.75,0.71
```

### Sample Loyalty Labels

```csv
customer_id,label,reference_date
C001,1,2024-01-20
C002,0,2024-01-20
```

---

## 22. UML Diagrams

This section contains UML diagrams that describe the system from multiple viewpoints. The diagrams are provided in PlantUML `.puml` source files under `report/diagrams/` so they can be edited and rendered into PNG/SVG as needed.

- **Use Case Diagram:** `report/diagrams/use_case.puml` — shows actors (stick figures) and primary use cases (ovals) for the Smart Loyalty System.
- **Sequence Diagram (Predict Loyalty):** `report/diagrams/sequence_predict_loyalty.puml` — shows the interaction flow between user, frontend, backend, model, and RFM data when making a loyalty prediction.
- **Architecture / Component Diagram:** `report/diagrams/architecture.puml` — high-level component view showing frontend, backend, model store, and data files.

Notation notes:

- Actors are represented as UML actors (stick figures in rendering).
- Use cases are represented as ovals inside the `Smart Loyalty System` boundary.
- Sequence lifelines and messages follow standard UML sequence diagram notation.

Rendering instructions (requires `plantuml` CLI or Docker image):

PowerShell (if `plantuml` is installed and on PATH):

```powershell
plantuml -tpng report/diagrams/use_case.puml
plantuml -tpng report/diagrams/sequence_predict_loyalty.puml
plantuml -tpng report/diagrams/architecture.puml
```

Docker (no local install):

```powershell
# from project root (Windows PowerShell)
docker run --rm -v ${PWD}:/workspace plantuml/plantuml -tpng /workspace/report/diagrams/use_case.puml
docker run --rm -v ${PWD}:/workspace plantuml/plantuml -tpng /workspace/report/diagrams/sequence_predict_loyalty.puml
docker run --rm -v ${PWD}:/workspace plantuml/plantuml -tpng /workspace/report/diagrams/architecture.puml
```

After rendering, the generated PNG files will be next to the `.puml` sources (e.g., `report/diagrams/use_case.png`).

---

**Document End**

_This SRS document provides a comprehensive specification of the Smart Loyalty System. Regular updates are recommended as the project evolves and new requirements emerge._

---

**Prepared for:** Academic Institution  
**Project Status:** Active Development  
**Next Review Date:** January 7, 2025
