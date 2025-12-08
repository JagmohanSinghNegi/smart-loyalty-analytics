from flask import Flask, jsonify, request, make_response
import os
import sys
from pathlib import Path
import pandas as pd
import joblib

# import basket recommender
try:
    from backend.utils import basket
except Exception:
    proj_root = Path(__file__).resolve().parents[1]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    from backend.utils import basket

app = Flask(__name__, static_folder='../frontend', static_url_path='/')

_MODEL = None
_PIPELINE = None
_RFM_DF = None


def _load_model_and_pipeline(model_path='models/loyalty_model.pkl', pipeline_path='models/pipeline.pkl'):
    global _MODEL, _PIPELINE
    if _MODEL is None and os.path.exists(model_path):
        _MODEL = joblib.load(model_path)
    if _PIPELINE is None and os.path.exists(pipeline_path):
        _PIPELINE = joblib.load(pipeline_path)
    return _MODEL, _PIPELINE


def _load_rfm(rfm_path='models/rfm_features.csv'):
    global _RFM_DF
    if _RFM_DF is None:
        if os.path.exists(rfm_path):
            _RFM_DF = pd.read_csv(rfm_path)
        else:
            _RFM_DF = pd.DataFrame()
    return _RFM_DF


@app.after_request
def _add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict-loyalty', methods=['POST'])
def predict_loyalty():
    if not request.is_json:
        return make_response(jsonify({'error': 'Request must be JSON'}), 400)
    payload = request.get_json()
    customer_id = payload.get('customer_id')
    if not customer_id:
        return make_response(jsonify({'error': 'Missing customer_id'}), 400)

    model, pipeline = _load_model_and_pipeline()
    if model is None:
        return make_response(jsonify({'error': 'Model not available'}), 500)

    rfm = _load_rfm()
    if rfm.empty:
        return make_response(jsonify({'error': 'RFM features not available'}), 500)

    # find customer - handle both string and float customer_ids
    try:
        cid_float = float(customer_id)
        row = rfm[rfm['customer_id'] == cid_float]
    except (ValueError, TypeError):
        row = rfm[rfm['customer_id'].astype(str) == str(customer_id)]
    
    if row.empty:
        return make_response(jsonify({'error': f'customer_id {customer_id} not found. Available IDs: {rfm["customer_id"].tolist()}'}), 400)
    X = row[['recency','frequency','monetary','rfm_score']]

    try:
        if pipeline is not None:
            proba = pipeline.predict_proba(X)[:,1]
        else:
            proba = model.predict_proba(X)[:,1]
        score = float(proba[0])
    except Exception:
        try:
            pred = model.predict(X)[0]
            score = float(pred)
        except Exception:
            return make_response(jsonify({'error':'prediction failed'}),500)

    return jsonify({'customer_id': customer_id, 'loyalty_score': score, 'loyal': score>=0.5})


@app.route('/recommend-products', methods=['POST'])
def recommend_products():
    if not request.is_json:
        return make_response(jsonify({'error': 'Request must be JSON'}), 400)
    payload = request.get_json()
    product = payload.get('product')
    top_n = payload.get('top_n',5)
    try:
        top_n = int(top_n)
    except Exception:
        return make_response(jsonify({'error':'top_n must be int'}),400)
    if not product:
        return make_response(jsonify({'error':'Missing product'}),400)
    recs = basket.recommend_for_product(product, top_n=top_n)
    return jsonify({'product': product, 'recommendations': recs})


if __name__ == '__main__':
    proj_root = Path(__file__).resolve().parents[1]
    os.chdir(proj_root)
    app.run(debug=True)

# if __name__ == "__main__":
#     import os
#     port = int(os.environ.get("PORT", 5000))  # Use Render assigned port
#     app.run(host="0.0.0.0", port=port) 
