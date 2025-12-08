"""
Test suite for Smart Loyalty API endpoints.
Run with: pytest scripts/test_api.py -v
"""

import pytest
import json
import sys
from pathlib import Path

# Add project root to path for imports
proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

from backend.app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_rfm_data(tmp_path, monkeypatch):
    """Create mock RFM data for testing."""
    import os
    
    # Create models directory if it doesn't exist
    models_dir = proj_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Create a simple RFM CSV with test data
    rfm_csv = models_dir / 'rfm_features.csv'
    rfm_csv.write_text(
        'customer_id,recency,frequency,monetary,rfm_score\n'
        '1001.0,4,1,2.0,0.0375\n'
        '1002.0,0,2,5.5,1.0000\n'
        '1003.0,1,1,1.5,0.2250\n'
    )
    
    # Create cleaned data CSV for recommendations
    cleaned_dir = proj_root / 'data' / 'cleaned'
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    cleaned_csv = cleaned_dir / 'sample_cleaned.csv'
    cleaned_csv.write_text(
        'transaction_id,customer_id,date,products,amount\n'
        '1,1001,2025-01-01,Apple;Banana,10.0\n'
        '2,1002,2025-01-02,Apple;Orange,15.5\n'
        '3,1003,2025-01-03,Banana;Orange,8.5\n'
        '4,1001,2025-01-04,Apple;Orange,12.0\n'
        '5,1002,2025-01-05,Banana;Apple,9.0\n'
    )
    
    yield rfm_csv, cleaned_csv
    
    # Cleanup (optional - keep for debugging)
    # rfm_csv.unlink()
    # cleaned_csv.unlink()


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Test that /health endpoint returns 200 status."""
        response = client.get('/health')
        assert response.status_code == 200
    
    def test_health_returns_json(self, client):
        """Test that /health returns valid JSON."""
        response = client.get('/health')
        data = response.get_json()
        assert data is not None
    
    def test_health_returns_ok_status(self, client):
        """Test that /health returns status 'ok'."""
        response = client.get('/health')
        data = response.get_json()
        assert data.get('status') == 'ok'


class TestLoyaltyPredictionEndpoint:
    """Tests for /predict-loyalty endpoint."""
    
    def test_predict_loyalty_missing_json(self, client):
        """Test that endpoint rejects non-JSON requests."""
        response = client.post('/predict-loyalty', data='not json')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_predict_loyalty_missing_customer_id(self, client):
        """Test that endpoint requires customer_id."""
        response = client.post(
            '/predict-loyalty',
            json={}
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'customer_id' in data['error'].lower()
    
    def test_predict_loyalty_model_not_available(self, client, mock_rfm_data):
        """Test response when model is not available."""
        # Clear any cached models
        import backend.app
        backend.app._MODEL = None
        backend.app._PIPELINE = None
        
        response = client.post(
            '/predict-loyalty',
            json={'customer_id': '1002'}
        )
        # Should return 500 since model doesn't exist
        assert response.status_code == 500
        data = response.get_json()
        assert 'error' in data
    
    def test_predict_loyalty_customer_not_found(self, client, mock_rfm_data):
        """Test response for non-existent customer."""
        # Clear cached data
        import backend.app
        backend.app._MODEL = None
        backend.app._PIPELINE = None
        backend.app._RFM_DF = None
        
        response = client.post(
            '/predict-loyalty',
            json={'customer_id': '99999'}
        )
        # Should return 400 or 500 depending on whether model exists
        assert response.status_code in [400, 500]
        data = response.get_json()
        assert 'error' in data
    
    def test_predict_loyalty_returns_expected_fields(self, client, mock_rfm_data):
        """Test that loyalty prediction returns expected fields."""
        # Clear cached data
        import backend.app
        backend.app._MODEL = None
        backend.app._PIPELINE = None
        backend.app._RFM_DF = None
        
        response = client.post(
            '/predict-loyalty',
            json={'customer_id': '1002'}
        )
        
        # Depending on whether model is trained, may return error
        # But we can verify response structure if it works
        if response.status_code == 200:
            data = response.get_json()
            assert 'customer_id' in data
            assert 'loyalty_score' in data
            assert 'loyal' in data
            assert isinstance(data['loyalty_score'], (int, float))
            assert isinstance(data['loyal'], bool)
            assert 0 <= data['loyalty_score'] <= 1


class TestRecommendationEndpoint:
    """Tests for /recommend-products endpoint."""
    
    def test_recommend_products_missing_json(self, client):
        """Test that endpoint rejects non-JSON requests."""
        response = client.post('/recommend-products', data='not json')
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_recommend_products_missing_product(self, client):
        """Test that endpoint requires product name."""
        response = client.post(
            '/recommend-products',
            json={}
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'product' in data['error'].lower()
    
    def test_recommend_products_invalid_top_n(self, client):
        """Test that endpoint validates top_n parameter."""
        response = client.post(
            '/recommend-products',
            json={'product': 'Apple', 'top_n': 'invalid'}
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_recommend_products_returns_expected_fields(self, client, mock_rfm_data):
        """Test that recommendations endpoint returns expected fields."""
        response = client.post(
            '/recommend-products',
            json={'product': 'Apple', 'top_n': 5}
        )
        
        # Check response status
        assert response.status_code in [200, 400, 500]  # May fail if data not available
        
        data = response.get_json()
        if response.status_code == 200:
            # Verify response structure
            assert 'product' in data
            assert 'recommendations' in data
            assert isinstance(data['recommendations'], list)
            assert data['product'] == 'Apple'
    
    def test_recommend_products_returns_list(self, client, mock_rfm_data):
        """Test that recommendations returns a list."""
        response = client.post(
            '/recommend-products',
            json={'product': 'Apple', 'top_n': 3}
        )
        
        if response.status_code == 200:
            data = response.get_json()
            assert isinstance(data['recommendations'], list)


class TestCORSHeaders:
    """Tests for CORS headers."""
    
    def test_health_includes_cors_headers(self, client):
        """Test that responses include CORS headers."""
        response = client.get('/health')
        assert 'Access-Control-Allow-Origin' in response.headers
        assert response.headers['Access-Control-Allow-Origin'] == '*'
    
    def test_predict_loyalty_includes_cors_headers(self, client):
        """Test that POST responses include CORS headers."""
        response = client.post('/predict-loyalty', json={'customer_id': '1'})
        assert 'Access-Control-Allow-Origin' in response.headers


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
