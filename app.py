from flask import Flask, request, jsonify
from flask_cors import CORS
from train_model import CibilScoreModel
import numpy as np
import os
import pandas as pd
import requests


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize model
model = CibilScoreModel()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

@app.route('/search_nearby_banks', methods=['GET'])
def search_nearby_banks():
    """Fetch nearby banks using Google Places API without CORS issues."""
    try:
        location = request.args.get('location')  # Expected format: "lat,lng"
        radius = request.args.get('radius', 5000)  # Default: 5000 meters (5 km)
        bank_name = request.args.get('bank_name', '')

        if not location:
            return jsonify({"status": "error", "message": "Location is required"}), 400

        # Google Places API URL
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": location,
            "radius": radius,
            "type": "bank",
            "keyword": bank_name,
            "key": GOOGLE_MAPS_API_KEY
        }

        # Make request to Google API
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, params=params)

        # Return response as JSON
        return jsonify(response.json())

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route('/predict_cibil', methods=['POST'])
def predict_cibil():
    try:
        data = request.json
        print("Received data:", data)
        
        pan_number = data.get('pan_number')
        if not pan_number:
            raise ValueError("PAN number is required")
            
        # Read CSV and find user
        df = pd.read_csv('cibil_data.csv')
        user_data = df[df['PAN'].str.strip().str.upper() == pan_number.strip().upper()]
        
        if user_data.empty:
            print(f"No data found for PAN: {pan_number}")
            base_score = 750
        else:
            base_score = int(user_data.iloc[0]['CIBIL'])  # Convert to int
            
        # New loan details
        loan_amount = float(data.get('loan_amount', 0))
        monthly_emi = float(data.get('monthly_emi', 0))
        on_time_payments = int(data.get('on_time_payments', 0))
        late_payments = int(data.get('late_payments', 0))
        
        print(f"Processing: Base Score={base_score}, Loan={loan_amount}, EMI={monthly_emi}")
        
        # More conservative score adjustments
        payment_history_score = (on_time_payments * 5) - (late_payments * 15)  # Reduced impact
        repayment_ratio = monthly_emi / loan_amount if loan_amount > 0 else 0
        
        # Calculate final score with smaller adjustments
        final_score = base_score
        
        # Add smaller bonuses/penalties
        if repayment_ratio > 0.15:
            final_score -= 6  # Reduced penalty
        if late_payments > 2:
            final_score -= 10  # Reduced penalty
        if on_time_payments >= 6:
            final_score += 5  # Reduced bonus
            
        # Add payment history score
        final_score += payment_history_score
            
        # Ensure score stays in valid range and close to base
        final_score = max(base_score - 50, min(base_score + 50, final_score))  # Limit change to ±50
        final_score = max(300, min(900, final_score))
        
        response_data = {
            'status': 'success',
            'predicted_score': int(final_score),
            'is_eligible': bool(final_score >= 700),  # Explicit conversion to bool
            'analysis': {
                'base_score': int(base_score),
                'payment_history': 'Excellent' if late_payments == 0 else 'Poor',
                'emi_ratio': f"{(repayment_ratio * 100):.1f}% of loan amount",
                'risk_level': 'Low' if late_payments == 0 and repayment_ratio <= 0.15 else 'Moderate'
            }
        }
        
        print("Sending response:", response_data)
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/train', methods=['GET'])
def train():
    try:
        # Train new model
        model.train()
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/initialize', methods=['GET'])
def initialize():
    try:
        # Check if model exists, if not train it
        if not os.path.exists('model/cibil_model.pkl'):
            model.train()
        else:
            model.load_model()
        
        return jsonify({
            'status': 'success',
            'message': 'Model initialized successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Initialize model on startup
    try:
        model.load_model()
    except:
        model.train()
    
    app.run(debug=True)