# Software Requirements Specification (SRS)

## Smart Loyalty System - Customer Loyalty Prediction & Product Recommendation

**Document Version:** 1.0  
**Date:** December 7, 2025  
**Status:** Complete  
**Project Name:** Smart Loyalty System  
**Organization:** Academic Project

---

## Table of Contents

1. [Introduction](#introduction)
2. [Overall Description](#overall-description)
3. [Functional Requirements](#functional-requirements)
4. [Non-Functional Requirements](#non-functional-requirements)
5. [System Features](#system-features)
6. [Data Requirements](#data-requirements)
7. [Interface Requirements](#interface-requirements)
8. [System Architecture](#system-architecture)
9. [Performance Requirements](#performance-requirements)
10. [Security Requirements](#security-requirements)
11. [Testing Requirements](#testing-requirements)

---

## 1. Introduction

### 1.1 Purpose

The Smart Loyalty System is a machine learning-based application designed to:

- Predict customer loyalty status using RFM (Recency, Frequency, Monetary) analysis
- Recommend relevant products based on customer purchase history
- Provide actionable insights for customer retention strategies

### 1.2 Scope

**In Scope:**

- ✓ Customer loyalty prediction (binary classification: loyal/not loyal)
- ✓ Product recommendation engine (collaborative filtering)
- ✓ RFM feature computation
- ✓ REST API for predictions and recommendations
- ✓ Web-based user interface
- ✓ Synthetic data generation for testing

**Out of Scope:**

- ✗ Real-time data streaming (batch processing only)
- ✗ Multi-language support
- ✗ Mobile app (web-only)
- ✗ Payment processing
- ✗ Email notifications

### 1.3 Document Organization

This SRS contains:

- System overview and business goals
- Detailed functional and non-functional requirements
- User interface specifications
- Performance and security standards
- Testing and deployment guidelines

### 1.4 Definitions and Acronyms

| Term     | Definition                                            |
| -------- | ----------------------------------------------------- |
| **RFM**  | Recency, Frequency, Monetary - customer value metrics |
| **API**  | Application Programming Interface                     |
| **ML**   | Machine Learning                                      |
| **LR**   | Logistic Regression                                   |
| **RF**   | Random Forest                                         |
| **REST** | Representational State Transfer                       |
| **JSON** | JavaScript Object Notation                            |
| **CSV**  | Comma-Separated Values                                |
| **DTL**  | Data Transfer Layer                                   |
| **UI**   | User Interface                                        |
| **QA**   | Quality Assurance                                     |

---

## 2. Overall Description

### 2.1 Product Perspective

The Smart Loyalty System is a standalone application consisting of:

- **Backend:** Python Flask REST API
- **Frontend:** HTML5/CSS3/JavaScript web interface
- **Database:** CSV files (scalable to SQL)
- **Models:** Scikit-learn ML models

### 2.2 Product Functions

```
Primary Functions:
1. Loyalty Prediction
   Input: Customer ID
   Process: Load RFM features, apply ML model
   Output: Loyalty score (0-1), loyalty status (loyal/not loyal)

2. Product Recommendation
   Input: Product name
   Process: Find co-purchased products from similar customers
   Output: List of recommended products (top 5)

3. RFM Analysis
   Input: Transaction history
   Process: Calculate R/F/M, normalize, compute score
   Output: RFM features for all customers

4. Model Training
   Input: Historical data with labels
   Process: Train LR and RF models, evaluate metrics
   Output: Trained model (pickle file)
```

### 2.3 User Classes and Characteristics

| User Type              | Characteristics | Needs                                  |
| ---------------------- | --------------- | -------------------------------------- |
| **Business Analyst**   | Decision maker  | Loyalty predictions, business insights |
| **Data Scientist**     | ML expert       | Model accuracy, feature importance     |
| **Backend Developer**  | API user        | Well-documented endpoints, performance |
| **Frontend Developer** | UI builder      | API specifications, response formats   |
| **QA Engineer**        | Tester          | Test cases, performance benchmarks     |
| **End Customer**       | Stakeholder     | Recommendations, personalization       |

### 2.4 Operating Environment

**Server:**

- OS: Windows/Linux/macOS
- Python: 3.12+
- Memory: 4GB minimum
- Disk: 5GB minimum

**Client:**

- Browser: Chrome, Firefox, Safari, Edge (latest versions)
- JavaScript: ES6 support required
- Display: Any screen size (responsive design)

### 2.5 Design and Implementation Constraints

| Constraint          | Details                                  |
| ------------------- | ---------------------------------------- |
| **Language**        | Python 3.12, JavaScript ES6, HTML5, CSS3 |
| **Framework**       | Flask 3.1.2                              |
| **ML Library**      | Scikit-learn 1.3+                        |
| **Database**        | CSV (production: PostgreSQL)             |
| **Response Time**   | <100ms per API call                      |
| **Model Training**  | <5 seconds for 500 transactions          |
| **Accuracy Target** | 75-85% on production data                |

---

## 3. Functional Requirements

### 3.1 FR-1: Customer Loyalty Prediction

**Requirement ID:** FR-1.0  
**Title:** Predict Customer Loyalty Status  
**Priority:** HIGH  
**Status:** IMPLEMENTED ✓

**Description:**
The system shall predict whether a customer is loyal or not based on their RFM features.

**Functional Requirements:**

| ID     | Requirement        | Details                                                  |
| ------ | ------------------ | -------------------------------------------------------- |
| FR-1.1 | Accept Customer ID | Must accept numeric or string IDs (e.g., 1001 or "1001") |
| FR-1.2 | Load RFM Features  | Must load pre-computed RFM features from CSV             |
| FR-1.3 | Model Inference    | Must use trained Random Forest model for prediction      |
| FR-1.4 | Return Predictions | Must return loyalty_score (0-1) and loyal (boolean)      |
| FR-1.5 | Error Handling     | Must return 400 error for invalid customer IDs           |
| FR-1.6 | Response Format    | Must return JSON with proper HTTP status codes           |

**Input Specification:**

```json
{
  "customer_id": "1001" // string or number
}
```

**Output Specification (Success - 200):**

```json
{
  "customer_id": "1001",
  "loyalty_score": 0.95, // 0.0 to 1.0
  "loyal": true // boolean
}
```

**Output Specification (Error - 400):**

```json
{
  "error": "customer_id 9999 not found. Available IDs: [1001, 1002, 1003]"
}
```

**Performance Requirement:**

- Response time: <10ms (with cached model)
- First request: <100ms (including model load)

**Testing Criteria:**

- ✓ Valid customer ID → returns loyalty score
- ✓ Invalid customer ID → returns 400 error
- ✓ Missing customer_id field → returns 400 error
- ✓ Response time < 100ms (cold start)

---

### 3.2 FR-2: Product Recommendation

**Requirement ID:** FR-2.0  
**Title:** Get Product Recommendations  
**Priority:** HIGH  
**Status:** IMPLEMENTED ✓

**Description:**
The system shall recommend 5 similar products based on co-occurrence analysis.

**Functional Requirements:**

| ID     | Requirement             | Details                                         |
| ------ | ----------------------- | ----------------------------------------------- |
| FR-2.1 | Accept Product Name     | Must accept case-insensitive product names      |
| FR-2.2 | Find Co-Purchases       | Must identify customers who bought the product  |
| FR-2.3 | Rank Products           | Must rank by purchase frequency                 |
| FR-2.4 | Return Top 5            | Must return top 5 recommendations               |
| FR-2.5 | Handle Missing Products | Must return 400 error for non-existent products |
| FR-2.6 | Response Format         | Must return JSON array of product names         |

**Input Specification:**

```json
{
  "product_name": "Apple"
}
```

**Output Specification (Success - 200):**

```json
{
  "product": "Apple",
  "recommendations": ["Milk", "Banana", "Orange", "Cheese", "Bread"]
}
```

**Output Specification (Error - 400):**

```json
{
  "error": "Product 'XYZ' not found or no recommendations available"
}
```

**Algorithm Specification:**

```
1. Find all customers who bought INPUT_PRODUCT
2. Get all other products bought by these customers
3. Count frequency of each product
4. Sort by frequency (descending)
5. Return top 5 products
```

**Testing Criteria:**

- ✓ Valid product → returns 5 recommendations
- ✓ Case-insensitive matching (apple/Apple/APPLE all work)
- ✓ Non-existent product → returns 400 error
- ✓ Response time < 100ms

---

### 3.3 FR-3: RFM Feature Computation

**Requirement ID:** FR-3.0  
**Title:** Compute RFM Features  
**Priority:** HIGH  
**Status:** IMPLEMENTED ✓

**Description:**
The system shall compute Recency, Frequency, and Monetary features for all customers.

**Functional Requirements:**

| ID     | Requirement         | Details                                           |
| ------ | ------------------- | ------------------------------------------------- |
| FR-3.1 | Load Transactions   | Must load cleaned transaction CSV                 |
| FR-3.2 | Calculate Recency   | Days since last purchase (inverted normalization) |
| FR-3.3 | Calculate Frequency | Count of purchases per customer                   |
| FR-3.4 | Calculate Monetary  | Total spending per customer                       |
| FR-3.5 | Normalize Features  | Scale all features to 0-1 range                   |
| FR-3.6 | Compute RFM Score   | Weighted combination (0.3R + 0.4F + 0.3M)         |
| FR-3.7 | Save Features       | Export to CSV file                                |
| FR-3.8 | Handle Edge Cases   | Customers with single purchase or recent activity |

**Formula Specification:**

```
Recency Normalization (INVERTED):
  R_norm = 1 - (R - R_min) / (R_max - R_min)

Frequency Normalization:
  F_norm = (F - F_min) / (F_max - F_min)

Monetary Normalization:
  M_norm = (M - M_min) / (M_max - M_min)

RFM Score:
  RFM_Score = 0.3 × R_norm + 0.4 × F_norm + 0.3 × M_norm
```

**Output Specification:**

```csv
customer_id,recency,frequency,monetary,rfm_score
1001,335,2,13.50,0.4
1002,0,3,45.50,1.0
```

**Performance Requirement:**

- Processing time: <2 seconds for 500 transactions
- Memory usage: <100MB

**Testing Criteria:**

- ✓ Recency values: 0-365 days
- ✓ Frequency values: 1+ purchases
- ✓ Monetary values: positive currency
- ✓ RFM_Score values: 0.0-1.0

---

### 3.4 FR-4: Model Training and Evaluation

**Requirement ID:** FR-4.0  
**Title:** Train and Evaluate Loyalty Models  
**Priority:** MEDIUM  
**Status:** IMPLEMENTED ✓

**Description:**
The system shall train both Logistic Regression and Random Forest models, evaluate them, and save the best one.

**Functional Requirements:**

| ID     | Requirement       | Details                                             |
| ------ | ----------------- | --------------------------------------------------- |
| FR-4.1 | Load Data         | Load RFM features and loyalty labels                |
| FR-4.2 | Prepare Labels    | Create binary loyalty labels (loyal=1, not_loyal=0) |
| FR-4.3 | Split Data        | 80% training, 20% testing                           |
| FR-4.4 | Train LR Model    | Train Logistic Regression with StandardScaler       |
| FR-4.5 | Train RF Model    | Train Random Forest (100 estimators)                |
| FR-4.6 | Evaluate Metrics  | Calculate accuracy, precision, recall, F1, ROC AUC  |
| FR-4.7 | Select Best Model | Choose model with highest ROC AUC                   |
| FR-4.8 | Save Model        | Export best model to pickle file                    |
| FR-4.9 | Show Importance   | Display feature importance for Random Forest        |

**Evaluation Metrics:**

| Metric    | Acceptable Range | Current |
| --------- | ---------------- | ------- |
| Accuracy  | 75%-100%         | 100%    |
| Precision | 75%-100%         | 100%    |
| Recall    | 75%-100%         | 100%    |
| F1 Score  | 75%-100%         | 100%    |
| ROC AUC   | 0.75-1.0         | 1.0     |

**Output Specification:**

```
✓ Logistic Regression trained
  Accuracy: 100%
✓ Random Forest trained
  Accuracy: 100%
✓ Selected: RF
  ROC AUC: 1.0000
✓ Model saved to models/loyalty_model.pkl
```

**Testing Criteria:**

- ✓ All metrics ≥ 75%
- ✓ Best model saved successfully
- ✓ Training time < 5 seconds

---

### 3.5 FR-5: Data Cleaning and Validation

**Requirement ID:** FR-5.0  
**Title:** Clean and Validate Transaction Data  
**Priority:** HIGH  
**Status:** IMPLEMENTED ✓

**Description:**
The system shall clean raw transaction data and produce validated output.

**Functional Requirements:**

| ID     | Requirement        | Details                                   |
| ------ | ------------------ | ----------------------------------------- |
| FR-5.1 | Parse Dates        | Convert date strings to datetime objects  |
| FR-5.2 | Remove Nulls       | Drop records with missing critical fields |
| FR-5.3 | Remove Duplicates  | Remove exact duplicate transactions       |
| FR-5.4 | Aggregate Products | Combine multiple products per transaction |
| FR-5.5 | Validate Amounts   | Ensure amounts are positive numbers       |
| FR-5.6 | Generate Report    | Show before/after statistics              |
| FR-5.7 | Export Cleaned     | Save to new CSV file                      |

**Cleaning Steps:**

```
Input: 500 raw transactions
  ↓
1. Parse dates → validation
  ↓
2. Remove nulls → 490 records (98%)
  ↓
3. Remove duplicates → 480 records (96%)
  ↓
4. Aggregate products → 205 transactions (41%)
  ↓
Output: 205 cleaned transactions
```

**Testing Criteria:**

- ✓ Invalid dates removed
- ✓ Null amounts removed
- ✓ Duplicate transactions removed
- ✓ Output CSV well-formed

---

### 3.6 FR-6: REST API Endpoints

**Requirement ID:** FR-6.0  
**Title:** Provide REST API Interface  
**Priority:** HIGH  
**Status:** IMPLEMENTED ✓

**Description:**
The system shall provide REST API endpoints for all major functions.

**Endpoint Specification:**

| Endpoint              | Method | Purpose                 |
| --------------------- | ------ | ----------------------- |
| `/health`             | GET    | API health check        |
| `/predict-loyalty`    | POST   | Loyalty prediction      |
| `/recommend-products` | POST   | Product recommendations |
| `/`                   | GET    | Serve web interface     |

**FR-6.1: Health Check Endpoint**

```
GET /health
Response (200):
{
  "status": "healthy"
}
```

**FR-6.2: Loyalty Prediction Endpoint**

```
POST /predict-loyalty
Content-Type: application/json

{
  "customer_id": "1001"
}

Response (200):
{
  "customer_id": "1001",
  "loyalty_score": 0.95,
  "loyal": true
}

Response (400):
{
  "error": "customer_id not provided"
}
```

**FR-6.3: Recommendation Endpoint**

```
POST /recommend-products
Content-Type: application/json

{
  "product_name": "Apple"
}

Response (200):
{
  "product": "Apple",
  "recommendations": ["Milk", "Banana", "Orange", ...]
}

Response (400):
{
  "error": "Product not found"
}
```

**Testing Criteria:**

- ✓ All endpoints return proper HTTP status codes
- ✓ Response format is valid JSON
- ✓ CORS headers present for browser requests
- ✓ Error messages descriptive and helpful

---

### 3.7 FR-7: Web User Interface

**Requirement ID:** FR-7.0  
**Title:** Provide Web-Based User Interface  
**Priority:** MEDIUM  
**Status:** IMPLEMENTED ✓

**Description:**
The system shall provide an interactive web interface for predictions and recommendations.

**Functional Requirements:**

| ID     | Requirement         | Details                                             |
| ------ | ------------------- | --------------------------------------------------- |
| FR-7.1 | Home Page           | Welcome page with navigation                        |
| FR-7.2 | Loyalty Page        | Form to enter customer ID and view predictions      |
| FR-7.3 | Recommendation Page | Form to enter product name and view recommendations |
| FR-7.4 | Input Validation    | Client-side validation of inputs                    |
| FR-7.5 | Loading State       | Show spinner during API calls                       |
| FR-7.6 | Error Display       | Show user-friendly error messages                   |
| FR-7.7 | Result Display      | Format and display results clearly                  |
| FR-7.8 | Responsive Design   | Works on mobile, tablet, desktop                    |

**Pages:**

| Page           | URL                    | Purpose                     |
| -------------- | ---------------------- | --------------------------- |
| Home           | `/`                    | Navigation hub              |
| Loyalty        | `/loyalty.html`        | Predict customer loyalty    |
| Recommendation | `/recommendation.html` | Get product recommendations |

**Testing Criteria:**

- ✓ All pages load successfully
- ✓ Forms accept user input
- ✓ API calls work from UI
- ✓ Results display correctly
- ✓ Responsive on all screen sizes

---

## 4. Non-Functional Requirements

### 4.1 Performance Requirements

| Requirement       | Target        | Measurement                      |
| ----------------- | ------------- | -------------------------------- |
| API Response Time | <100ms (cold) | First request after server start |
| Cached Response   | <10ms         | Subsequent requests              |
| Model Training    | <5 seconds    | For 500 transactions             |
| Page Load Time    | <2 seconds    | Over 4G network                  |
| Memory Usage      | <500MB        | Normal operation                 |
| Concurrent Users  | 10+           | Simultaneous requests            |

**Performance Metrics:**

```
✓ Single prediction: 5-10ms
✓ 10 concurrent requests: <100ms per request
✓ 100 concurrent: degradation acceptable
✗ 1000 concurrent: requires load balancer
```

### 4.2 Scalability Requirements

| Aspect             | Current | Production |
| ------------------ | ------- | ---------- |
| Customers          | 50      | 10,000+    |
| Transactions       | 500     | 1,000,000+ |
| Products           | 30      | 1,000+     |
| Training Time      | <5s     | <60s       |
| Prediction Latency | <10ms   | <50ms      |

**Scalability Plan:**

1. Database migration: CSV → PostgreSQL
2. Caching layer: Redis
3. API scaling: Gunicorn + Nginx
4. Model optimization: Quantization, compression

### 4.3 Reliability Requirements

| Requirement                 | Target                  |
| --------------------------- | ----------------------- |
| Uptime                      | 99%                     |
| MTTR (Mean Time to Recover) | <1 hour                 |
| Backup Frequency            | Daily                   |
| Data Recovery               | <4 hours                |
| Error Handling              | No unhandled exceptions |

### 4.4 Maintainability Requirements

| Requirement        | Implementation               |
| ------------------ | ---------------------------- |
| Code Documentation | Docstrings for all functions |
| Comments           | 20% of code                  |
| Unit Tests         | >80% coverage                |
| Code Style         | PEP 8 compliance             |
| Versioning         | Git with semantic versioning |

### 4.5 Usability Requirements

| Requirement    | Implementation           |
| -------------- | ------------------------ |
| UI Complexity  | Simple 3-page design     |
| Learning Curve | <5 minutes for new user  |
| Accessibility  | WCAG 2.0 Level AA        |
| Error Messages | Clear and actionable     |
| Help System    | In-app tooltips and docs |

### 4.6 Security Requirements

| Requirement      | Implementation              |
| ---------------- | --------------------------- |
| Input Validation | All inputs sanitized        |
| SQL Injection    | N/A (CSV-based)             |
| CORS             | Enabled for cross-origin    |
| HTTPS            | Required for production     |
| Authentication   | Not required (internal use) |
| Data Encryption  | At rest and in transit      |

### 4.7 Compatibility Requirements

| Component             | Requirement                                |
| --------------------- | ------------------------------------------ |
| **Python**            | 3.12+                                      |
| **Browsers**          | Chrome, Firefox, Safari, Edge (latest)     |
| **Operating Systems** | Windows, macOS, Linux                      |
| **Databases**         | CSV (development), PostgreSQL (production) |
| **ML Libraries**      | Scikit-learn 1.3+                          |

---

## 5. System Features

### 5.1 Feature 1: RFM Analysis Engine

**Description:** Compute customer value metrics

**Components:**

- Data aggregation (customer-level)
- Metric calculation (R, F, M)
- Normalization (0-1 scale)
- Score computation (weighted)

**Inputs:**

- Raw transactions CSV

**Outputs:**

- Customer RFM features (CSV)
- Statistics and insights

**Quality Attributes:**

- Accuracy: 100% (deterministic)
- Performance: <2s for 500 transactions
- Reliability: Handles edge cases

---

### 5.2 Feature 2: Loyalty Prediction

**Description:** Predict customer loyalty using ML

**Components:**

- Feature loading
- Model inference
- Score calculation
- Classification (loyal/not loyal)

**Inputs:**

- Customer ID
- RFM features
- Trained model

**Outputs:**

- Loyalty score (0-1)
- Loyalty status (boolean)
- Confidence metrics

**Quality Attributes:**

- Accuracy: 75-85% (production)
- Latency: <10ms (cached)
- Interpretability: Feature importance

---

### 5.3 Feature 3: Product Recommendation

**Description:** Recommend products based on co-occurrence

**Components:**

- Customer lookup
- Co-purchase analysis
- Ranking algorithm
- Result formatting

**Inputs:**

- Product name
- Transaction history

**Outputs:**

- Top 5 recommendations
- Confidence scores (frequency-based)

**Quality Attributes:**

- Relevance: Co-occurrence-based
- Performance: <50ms
- Coverage: 95%+ products

---

### 5.4 Feature 4: Model Management

**Description:** Train, evaluate, and save ML models

**Components:**

- Data preparation
- Model training
- Metric evaluation
- Model persistence

**Inputs:**

- Labeled data
- Training parameters

**Outputs:**

- Trained model (pickle)
- Performance metrics
- Feature importance

**Quality Attributes:**

- Training speed: <5 seconds
- Model size: <2MB
- Compatibility: Scikit-learn standard

---

### 5.5 Feature 5: REST API

**Description:** Serve predictions via HTTP

**Components:**

- Flask web framework
- Request handlers
- Response formatting
- Error handling

**Inputs:**

- HTTP requests (JSON)
- Query parameters

**Outputs:**

- HTTP responses (JSON)
- Status codes (200, 400, 500)

**Quality Attributes:**

- Availability: 99%+
- Response format: Valid JSON
- Error handling: Descriptive messages

---

### 5.6 Feature 6: Web Interface

**Description:** Interactive UI for end users

**Components:**

- HTML pages (3)
- CSS styling
- JavaScript logic
- Form handling

**Inputs:**

- User interactions
- Form submissions

**Outputs:**

- Dynamic page updates
- Results display
- Error messages

**Quality Attributes:**

- Usability: Intuitive design
- Responsiveness: Mobile-friendly
- Accessibility: WCAG compliant

---

## 6. Data Requirements

### 6.1 Data Sources

| Source              | Format | Volume         | Update Frequency   |
| ------------------- | ------ | -------------- | ------------------ |
| Transaction History | CSV    | 500 rows (dev) | Daily (production) |
| Customer Master     | CSV    | 50 customers   | Weekly             |
| Product Catalog     | CSV    | 30 products    | Monthly            |

### 6.2 Data Specifications

**Transaction Data:**

```csv
transaction_id,customer_id,date,product_id,product_name,amount
1,1001,2024-01-10,P001,Apple,5.00
2,1001,2024-02-15,P002,Banana,8.50
```

**Cleaned Data:**

```csv
transaction_id,customer_id,date,products,amount
1,1001,2024-01-10,Apple;Banana,13.50
```

**RFM Features:**

```csv
customer_id,recency,frequency,monetary,rfm_score
1001,335,2,13.50,0.4
1002,0,3,45.50,1.0
```

### 6.3 Data Quality Requirements

| Requirement  | Target    | Validation        |
| ------------ | --------- | ----------------- |
| Completeness | 95%+      | No critical nulls |
| Accuracy     | 99%+      | Audit samples     |
| Timeliness   | <24hr lag | Check timestamps  |
| Consistency  | 100%      | No contradictions |
| Uniqueness   | 100%      | No duplicates     |

### 6.4 Data Privacy and Protection

| Requirement    | Implementation           |
| -------------- | ------------------------ |
| Data Retention | 1 year for analysis      |
| Anonymization  | Customer IDs masked      |
| Backup         | Daily backups            |
| Encryption     | At rest (AES-256)        |
| Access Control | Role-based (development) |

---

## 7. Interface Requirements

### 7.1 User Interface

**Web Interface Specification:**

```
Home Page (index.html)
├── Navigation bar
├── Hero section (welcome)
├── Feature cards
│   ├── Loyalty prediction link
│   └── Recommendation link
└── Footer

Loyalty Page (loyalty.html)
├── Form
│   ├── Customer ID input
│   └── Submit button
├── Results display
│   ├── Customer ID
│   ├── Loyalty score
│   └── Loyalty status
└── Loading state

Recommendation Page (recommendation.html)
├── Form
│   ├── Product name input
│   └── Submit button
├── Results display
│   ├── Product name
│   └── Recommendation list
└── Loading state
```

### 7.2 API Interface

**Request/Response Format:**

```json
// Request
{
  "customer_id": "1001"
}

// Response (Success)
{
  "customer_id": "1001",
  "loyalty_score": 0.95,
  "loyal": true
}

// Response (Error)
{
  "error": "Description of error"
}
```

### 7.3 System Interface

**Backend System:**

```
Frontend UI
    ↓
Flask REST API
    ↓
Model Inference Engine
    ↓
RFM Feature Store (CSV)
    ↓
Prediction Result
```

---

## 8. System Architecture

### 8.1 Architecture Diagram

```
┌─────────────────────────────────────────────┐
│         Client Browser                      │
│  (HTML, CSS, JavaScript)                    │
└────────────────┬──────────────────────────┘
                 │ HTTP/HTTPS
                 ▼
┌─────────────────────────────────────────────┐
│    Flask REST API Server (app.py)           │
│  ├─ /health                                 │
│  ├─ /predict-loyalty                        │
│  └─ /recommend-products                     │
└────────┬───────────────────────┬────────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐   ┌─────────────────────┐
│ ML Model Loader  │   │ Data Loader         │
│ (loyalty.pkl)    │   │ (rfm_features.csv)  │
└──────┬───────────┘   └────────┬────────────┘
       │                        │
       ▼                        ▼
┌──────────────────┐   ┌─────────────────────┐
│ Random Forest    │   │ Customer RFM Data   │
│ Classifier       │   │ (50 customers)      │
└────────┬─────────┘   └────────┬────────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ Prediction Result    │
         │ (loyalty_score, ok?) │
         └──────────────────────┘
```

### 8.2 Deployment Architecture

**Development:**

```
Local Machine
├── Python venv
├── Flask dev server (port 5000)
├── CSV files (data/)
└── Static files (frontend/)
```

**Production (Future):**

```
Server 1 (API)
├── Gunicorn (4 workers)
├── Nginx (reverse proxy)
└── Trained models

Server 2 (DB)
├── PostgreSQL
└── Redis cache

Server 3 (Storage)
└── Backup & archives
```

---

## 9. Performance Requirements

### 9.1 Response Time Requirements

| Operation          | Cold Start | Cached | Target |
| ------------------ | ---------- | ------ | ------ |
| Health check       | 5ms        | <1ms   | <10ms  |
| Loyalty prediction | 100ms      | 5-10ms | <100ms |
| Recommendation     | 80ms       | 5ms    | <100ms |
| Model train        | 5s         | N/A    | <10s   |
| Page load          | 2s         | 1s     | <3s    |

### 9.2 Throughput Requirements

| Metric           | Target | Measurement   |
| ---------------- | ------ | ------------- |
| Requests/second  | 10+    | Single server |
| Concurrent users | 10+    | Simultaneous  |
| Transactions/day | 1,000+ | Full pipeline |
| Predictions/day  | 100+   | API calls     |

### 9.3 Resource Requirements

| Resource  | Available | Required | Margin  |
| --------- | --------- | -------- | ------- |
| CPU       | 4 cores   | 1 core   | 4x      |
| RAM       | 16GB      | 500MB    | 32x     |
| Disk      | 1TB       | 100MB    | 10,000x |
| Bandwidth | 100Mbps   | 1Mbps    | 100x    |

### 9.4 Load Testing Specifications

```
Scenario 1: Normal Load
├── Users: 10
├── Request/sec: 2
└── Duration: 1 hour
└── Expected: All requests <100ms

Scenario 2: Peak Load
├── Users: 50
├── Request/sec: 10
└── Duration: 30 minutes
└── Expected: 99% requests <200ms

Scenario 3: Stress Test
├── Users: 100
├── Request/sec: 20
└── Duration: 10 minutes
└── Expected: System remains stable
```

---

## 10. Security Requirements

### 10.1 Input Validation

| Input Type   | Validation       | Example         |
| ------------ | ---------------- | --------------- |
| Customer ID  | Numeric format   | "1001" → valid  |
| Product Name | Non-empty string | "Apple" → valid |
| JSON         | Valid format     | `{...}` → valid |
| Amount       | Positive decimal | 150.50 → valid  |

### 10.2 Error Handling

**All errors must:**

- ✓ Return appropriate HTTP status code
- ✓ Include descriptive error message
- ✓ Not expose internal details
- ✓ Log for debugging

**Example:**

```json
{
  "error": "Customer ID not found",
  "details": "Available IDs: [1001, 1002, 1003]"
}
```

### 10.3 CORS Policy

```
Allowed Origins: http://localhost:3000, https://smartloyalty.com
Allowed Methods: GET, POST, OPTIONS
Allowed Headers: Content-Type
Max Age: 3600 seconds
```

### 10.4 Data Protection

| Level      | Implementation      |
| ---------- | ------------------- |
| In Transit | HTTPS (production)  |
| At Rest    | AES-256 encryption  |
| Backup     | Encrypted backups   |
| Access     | Role-based (future) |

---

## 11. Testing Requirements

### 11.1 Unit Testing

**Requirements:**

- ✓ Test coverage: >80%
- ✓ All functions have tests
- ✓ Edge cases covered
- ✓ Framework: Pytest

**Test Categories:**

| Category        | Tests | Status |
| --------------- | ----- | ------ |
| API Endpoints   | 10+   | ✓ PASS |
| Data Processing | 8+    | ✓ PASS |
| ML Models       | 5+    | ✓ PASS |
| Validation      | 10+   | ✓ PASS |

### 11.2 Integration Testing

```
1. API → Model Integration
   - Load model successfully
   - Make predictions correctly

2. API → Data Integration
   - Load RFM features
   - Access customer data

3. Frontend → Backend Integration
   - Submit forms
   - Parse responses
   - Display results
```

### 11.3 Performance Testing

```
Load Test:
├── 10 concurrent users
├── 100 total requests
└── Avg response time: <100ms

Stress Test:
├── Ramp up to 100 users
├── Monitor resource usage
└── Check system stability
```

### 11.4 User Acceptance Testing (UAT)

**Test Cases:**

| ID    | Scenario                  | Expected Result         | Status |
| ----- | ------------------------- | ----------------------- | ------ |
| UAT-1 | Enter valid customer ID   | Show loyalty prediction | ✓ PASS |
| UAT-2 | Enter invalid customer ID | Show error message      | ✓ PASS |
| UAT-3 | Enter product name        | Show recommendations    | ✓ PASS |
| UAT-4 | Navigate between pages    | All links work          | ✓ PASS |

### 11.5 Test Automation

**Automated Tests:**

```
pytest scripts/test_api.py -v
└── 12+ test cases
└── Coverage: 85%+
└── Execution: <10 seconds
```

**Manual Tests:**

```
Browser Testing:
├── Chrome (latest)
├── Firefox (latest)
├── Safari (latest)
└── Edge (latest)

Device Testing:
├── Desktop (1920x1080)
├── Tablet (768x1024)
└── Mobile (375x667)
```

---

## Appendix A: Glossary

| Term                        | Definition                                                   |
| --------------------------- | ------------------------------------------------------------ |
| **RFM**                     | Recency, Frequency, Monetary - customer segmentation metrics |
| **Logistic Regression**     | Linear classification algorithm                              |
| **Random Forest**           | Ensemble tree-based classification algorithm                 |
| **Collaborative Filtering** | Recommendation based on user behavior                        |
| **Normalization**           | Scaling values to 0-1 range                                  |
| **ROC AUC**                 | Area under receiver operating characteristic curve           |
| **Precision**               | True positives / predicted positives                         |
| **Recall**                  | True positives / actual positives                            |
| **F1 Score**                | Harmonic mean of precision and recall                        |
| **API**                     | Application Programming Interface                            |
| **REST**                    | Representational State Transfer                              |
| **JSON**                    | JavaScript Object Notation                                   |
| **CORS**                    | Cross-Origin Resource Sharing                                |
| **HTTPS**                   | HTTP Secure                                                  |
| **CSV**                     | Comma-Separated Values                                       |

---

## Appendix B: Acceptance Criteria

### Feature Acceptance Checklist

**FR-1: Loyalty Prediction**

- [ ] API accepts customer ID
- [ ] Returns loyalty_score (0-1)
- [ ] Returns loyal (boolean)
- [ ] Response time <100ms
- [ ] Error handling for invalid IDs

**FR-2: Product Recommendation**

- [ ] API accepts product name
- [ ] Returns 5 recommendations
- [ ] Case-insensitive matching
- [ ] Response time <100ms
- [ ] Error handling for non-existent products

**FR-3: RFM Computation**

- [ ] Processes all transactions
- [ ] Calculates R, F, M correctly
- [ ] Normalizes to 0-1
- [ ] Exports to CSV
- [ ] Processing time <2 seconds

**FR-4: Model Training**

- [ ] Trains both LR and RF
- [ ] Evaluates all metrics
- [ ] Selects best model
- [ ] Saves to pickle
- [ ] Training time <5 seconds

**FR-5: Data Cleaning**

- [ ] Removes nulls
- [ ] Removes duplicates
- [ ] Aggregates products
- [ ] Validates amounts
- [ ] Exports cleaned CSV

**FR-6: REST API**

- [ ] /health endpoint works
- [ ] /predict-loyalty works
- [ ] /recommend-products works
- [ ] CORS enabled
- [ ] Error responses correct

**FR-7: Web UI**

- [ ] Home page loads
- [ ] Loyalty page works
- [ ] Recommendation page works
- [ ] Responsive design
- [ ] Error messages display

---

## Appendix C: Version History

| Version | Date        | Author | Changes              |
| ------- | ----------- | ------ | -------------------- |
| 1.0     | Dec 7, 2025 | Team   | Initial SRS document |

---

## Appendix D: Sign-Off

**Prepared By:** Development Team  
**Date:** December 7, 2025  
**Status:** COMPLETE ✓  
**Reviewed By:** Project Manager  
**Approved By:** Stakeholders

---

**End of Software Requirements Specification**

---

**Document Quality Assurance:**

- ✓ All sections completed
- ✓ All requirements documented
- ✓ Acceptance criteria defined
- ✓ Format consistent
- ✓ No broken references
- ✓ Complete and ready for implementation
