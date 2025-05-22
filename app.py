from flask import Flask, request, jsonify
from flask_cors import CORS
from train_model import CibilScoreModel
import numpy as np
import os
import pandas as pd
from pymongo import MongoClient
from bson.objectid import ObjectId
import hashlib
import uuid
import datetime
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Connect to MongoDB
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB = os.getenv('MONGODB_DB')

if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable is not set")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]
cibil_collection = db['cibil_data']
users_collection = db['users']
education_collection = db['education']
loan_collection = db['loan_applications']

# Initialize model
model = CibilScoreModel()

# Load models
model_dir = os.path.dirname(os.path.abspath(__file__))

# Load CIBIL prediction model
try:
    cibil_model = joblib.load(os.path.join(model_dir, 'cibil_model.pkl'))
    cibil_scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    print("CIBIL prediction model loaded successfully")
except Exception as e:
    print(f"Error loading CIBIL model: {e}")
    cibil_model = None
    cibil_scaler = None
    
# Load loan default prediction model
try:
    default_model = joblib.load(os.path.join(model_dir, 'default_model.pkl'))
    default_scaler = joblib.load(os.path.join(model_dir, 'default_scaler.pkl'))
    print("Loan default prediction model loaded successfully")
except Exception as e:
    print(f"Error loading default model: {e}")
    default_model = None
    default_scaler = None

# Helper functions for authentication
def hash_password(password, salt=None):
    """Hash a password for storing."""
    if salt is None:
        salt = uuid.uuid4().hex
    
    hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
    return {'salt': salt, 'hashed': hashed_password}

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password.get('salt')
    stored_hash = stored_password.get('hashed')
    verification_hash = hashlib.sha256((provided_password + salt).encode()).hexdigest()
    return stored_hash == verification_hash

@app.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    try:
        data = request.json
        
        # Validate required fields
        if not all(key in data for key in ['email', 'password', 'pan_number']):
            return jsonify({
                'status': 'error',
                'message': 'Email, password and PAN number are required'
            }), 400
        
        # Check if user already exists
        existing_user = users_collection.find_one({'email': data['email']})
        if existing_user:
            return jsonify({
                'status': 'error',
                'message': 'Email already registered'
            }), 400
        
        # Check if PAN is already registered
        existing_pan = users_collection.find_one({'pan_number': data['pan_number'].strip().upper()})
        if existing_pan:
            return jsonify({
                'status': 'error',
                'message': 'PAN number already registered'
            }), 400
        
        # Hash the password
        password_data = hash_password(data['password'])
        
        # Create user document
        user = {
            'email': data['email'],
            'password': password_data,  # Store hash and salt
            'pan_number': data['pan_number'].strip().upper(),
            'name': data.get('name', ''),
            'created_at': datetime.datetime.utcnow(),
            'last_login': None
        }
        
        # Insert into database
        result = users_collection.insert_one(user)
        
        # Return success without password
        user_id = str(result.inserted_id)
        return jsonify({
            'status': 'success',
            'message': 'User registered successfully',
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/login', methods=['POST'])
def login():
    """Authenticate a user."""
    try:
        data = request.json
        
        # Validate required fields
        if not all(key in data for key in ['email', 'password']):
            return jsonify({
                'status': 'error',
                'message': 'Email and password are required'
            }), 400
        
        # Find the user
        user = users_collection.find_one({'email': data['email']})
        
        if not user:
            return jsonify({
                'status': 'error',
                'message': 'User not found'
            }), 404
        
        # Verify password
        if not verify_password(user['password'], data['password']):
            return jsonify({
                'status': 'error',
                'message': 'Invalid credentials'
            }), 401
        
        # Update last login time
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.datetime.utcnow()}}
        )
        
        # Return user data (excluding password)
        user_data = {
            'user_id': str(user['_id']),
            'email': user['email'],
            'name': user.get('name', ''),
            'pan_number': user.get('pan_number', '')
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'user': user_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict_cibil', methods=['POST'])
def predict_cibil():
    try:
        data = request.json
        print("Received data:", data)
        
        pan_number = data.get('pan_number')
        if not pan_number:
            raise ValueError("PAN number is required")
            
        # Find user in MongoDB
        user_data = cibil_collection.find_one({'PAN': pan_number.strip().upper()})
        
        if not user_data:
            print(f"No data found for PAN: {pan_number}")
            base_score = float(data.get('current_score', 300))
        else:
            base_score = float(user_data.get('CIBIL', 300))
            
        try:
            # Get all required parameters with defaults
            loan_amount = float(data.get('loan_amount', 0))
            tenure_months = float(data.get('tenure_months', 0))
            late_payments = int(data.get('late_payments', 0))
            monthly_emi = float(data.get('monthly_emi', 0))
            credit_utilization = float(data.get('credit_utilization', 30))  # Default 30%
            credit_age_months = float(data.get('credit_age_months', 24))  # Default 2 years
            total_accounts = int(data.get('total_accounts', 1))  # Default 1 account
            credit_mix_types = int(data.get('credit_mix_types', 1))  # Default 1 type
            recent_inquiries = int(data.get('recent_inquiries', 0))  # Default 0 inquiries
            monthly_income = float(data.get('monthly_income', 50000))  # Default 50,000
            
            # Handle edge cases
            if monthly_emi <= 0 and loan_amount > 0 and tenure_months > 0:
                # Calculate a reasonable EMI if not provided
                interest_rate = 10.0  # Assume 10% interest rate
                monthly_rate = interest_rate / (12 * 100)
                monthly_emi = (loan_amount * monthly_rate * pow((1 + monthly_rate), tenure_months)) / (pow((1 + monthly_rate), tenure_months) - 1)
                print(f"Calculated monthly EMI: {monthly_emi}")
                
        except (ValueError, TypeError) as e:
            print(f"Error with numeric values: {e}")
            raise ValueError(f"Invalid numeric values provided: {e}")

        # 1. Payment History Impact (35% weightage)
        payment_history_score = 0
        if late_payments == 0:
            payment_history_score = 35  # Full score for perfect payment history
        elif late_payments == 1:
            payment_history_score = 25  # Single late payment
        elif late_payments == 2:
            payment_history_score = 15  # Two late payments
        else:
            payment_history_score = max(0, 35 - (late_payments * 10))  # Severe impact for multiple late payments

        # 2. Credit Utilization Impact (30% weightage)
        utilization_score = 0
        if credit_utilization <= 30:
            utilization_score = 30  # Ideal utilization
        elif credit_utilization <= 50:
            utilization_score = 20  # Moderate utilization
        elif credit_utilization <= 70:
            utilization_score = 10  # High utilization
        else:
            utilization_score = 5   # Very high utilization

        # 3. Credit Age Impact (15% weightage)
        age_score = 0
        if credit_age_months >= 60:  # 5+ years
            age_score = 15
        elif credit_age_months >= 36:  # 3+ years
            age_score = 12
        elif credit_age_months >= 24:  # 2+ years
            age_score = 8
        else:
            age_score = 5  # Less than 2 years

        # 4. Credit Mix Impact (10% weightage)
        mix_score = 0
        if credit_mix_types >= 3:
            mix_score = 10  # Diverse credit mix
        elif credit_mix_types == 2:
            mix_score = 7   # Moderate mix
        else:
            mix_score = 5   # Limited mix

        # 5. Credit Inquiries Impact (10% weightage)
        inquiry_score = 0
        if recent_inquiries == 0:
            inquiry_score = 10  # No recent inquiries
        elif recent_inquiries == 1:
            inquiry_score = 7   # One inquiry
        elif recent_inquiries == 2:
            inquiry_score = 5   # Two inquiries
        else:
            inquiry_score = max(0, 10 - (recent_inquiries * 3))  # Multiple inquiries

        # Calculate total score components
        total_factor_score = (payment_history_score + utilization_score + 
                            age_score + mix_score + inquiry_score)
        
        # Calculate base score range (300-900)
        score_range = 600  # (900 - 300)
        min_score = 300
        
        # Convert factor score (max 100) to CIBIL range
        calculated_score = min_score + ((total_factor_score / 100) * score_range)
        
        # Apply EMI to Income Ratio Impact
        if monthly_income > 0 and monthly_emi > 0:
            emi_to_income = (monthly_emi / monthly_income) * 100
            print(f"EMI to Income Ratio: {emi_to_income}%")
            
            if emi_to_income > 50:
                calculated_score -= 30  # High EMI burden
            elif emi_to_income > 40:
                calculated_score -= 20  # Moderate EMI burden
            elif emi_to_income > 30:
                calculated_score -= 10  # Acceptable EMI burden
        else:
            print("Skipping EMI to income ratio (zero values)")

        # Ensure score stays within valid range
        final_score = max(300, min(900, int(calculated_score)))
        
        # Calculate improvement potential
        improvement_potential = {
            'payment_history': int(35 - payment_history_score),
            'credit_utilization': int(30 - utilization_score),
            'credit_age': int(15 - age_score),
            'credit_mix': int(10 - mix_score),
            'recent_inquiries': int(10 - inquiry_score)
        }

        # Generate recovery timeline
        recovery_timeline = []
        current_score = final_score
        
        for month in range(3, 25, 3):
            score_improvement = 0
            
            # Payment history improvement
            if late_payments > 0:
                score_improvement += 5  # Improvement from consistent payments
            
            # Credit utilization improvement
            if credit_utilization > 30:
                score_improvement += 10  # Assuming reduction in utilization
            
            # Credit age improvement
            if credit_age_months < 60:
                score_improvement += 2  # Natural improvement with time
            
            # Apply improvement
            current_score = min(900, current_score + score_improvement)
            
            recovery_timeline.append({
                'month': month,
                'score': int(current_score),
                'improvement': int(current_score - final_score)
            })

        # Debug output
        print(f"Base score: {base_score}, Final score: {final_score}")
        print(f"Score components: Payment={payment_history_score}, Utilization={utilization_score}, Age={age_score}, Mix={mix_score}, Inquiries={inquiry_score}")
        print(f"Total factor score: {total_factor_score}/100 => {calculated_score} CIBIL")

        response_data = {
            'status': 'success',
            'starting_score': int(base_score),
            'calculated_score': final_score,
            'predicted_score': final_score,  # For backward compatibility
            'score_components': {
                'payment_history': {
                    'score': int(payment_history_score),
                    'max_score': 35,
                    'percentage': round((payment_history_score/35) * 100, 1)
                },
                'credit_utilization': {
                    'score': int(utilization_score),
                    'max_score': 30,
                    'percentage': round((utilization_score/30) * 100, 1)
                },
                'credit_age': {
                    'score': int(age_score),
                    'max_score': 15,
                    'percentage': round((age_score/15) * 100, 1)
                },
                'credit_mix': {
                    'score': int(mix_score),
                    'max_score': 10,
                    'percentage': round((mix_score/10) * 100, 1)
                },
                'recent_inquiries': {
                    'score': int(inquiry_score),
                    'max_score': 10,
                    'percentage': round((inquiry_score/10) * 100, 1)
                }
            },
            'improvement_potential': improvement_potential,
            'recovery_timeline': recovery_timeline,
            'risk_level': 'High' if final_score < 650 else ('Moderate' if final_score < 750 else 'Low'),
            'is_eligible': final_score >= 700,
            'recommendations': [
                "Ensure timely payment of all EMIs and credit card bills",
                "Keep credit utilization below 30%",
                "Avoid multiple loan applications in short periods",
                "Maintain a healthy mix of credit types",
                "Build longer credit history through consistent credit behavior"
            ]
        }
        
        # Store prediction in MongoDB
        loan_collection.insert_one({
            'pan_number': pan_number,
            'starting_score': int(base_score),
            'calculated_score': final_score,
            'improvement': int(final_score - base_score),
            'loan_amount': loan_amount,
            'tenure_months': int(tenure_months),
            'monthly_emi': monthly_emi,
            'late_payments': late_payments,
            'credit_utilization': credit_utilization,
            'risk_level': response_data['risk_level'],
            'created_at': pd.Timestamp.now().isoformat()
        })
        
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

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify the server is running."""
    return jsonify({
        'status': 'success',
        'message': 'Flask server is running',
        'mongo_connection': 'active' if client.server_info() else 'inactive',
        'ml_models': {
            'cibil_model': 'loaded' if cibil_model else 'not_loaded',
            'default_model': 'loaded' if default_model else 'not_loaded'
        }
    })

@app.route('/users', methods=['GET'])
def get_users():
    """Get all users from MongoDB."""
    try:
        users = list(users_collection.find({}, {'password': 0}))  # Exclude passwords
        # Convert ObjectId to string for JSON serialization
        for user in users:
            user['_id'] = str(user['_id'])
        return jsonify({
            'status': 'success', 
            'users': users
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/cibil_data', methods=['GET'])
def get_cibil_data():
    """Get CIBIL data, optionally filtered by PAN number."""
    try:
        pan = request.args.get('pan')
        filter_query = {'PAN': pan.upper()} if pan else {}
        
        cibil_data = list(cibil_collection.find(filter_query))
        # Convert ObjectId to string for JSON serialization
        for data in cibil_data:
            data['_id'] = str(data['_id'])
            
        return jsonify({
            'status': 'success',
            'data': cibil_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/import_csv', methods=['GET'])
def import_csv_to_mongodb():
    """Import initial CSV data to MongoDB."""
    try:
        # Check if collection is empty first
        if cibil_collection.count_documents({}) == 0:
            # Import from CSV
            df = pd.read_csv('cibil_data.csv')
            records = df.to_dict('records')
            cibil_collection.insert_many(records)
            return jsonify({
                'status': 'success',
                'message': f'Imported {len(records)} records to MongoDB'
            })
        else:
            return jsonify({
                'status': 'success',
                'message': 'Data already exists in MongoDB'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/education', methods=['GET', 'POST', 'DELETE'])
def education():
    if request.method == 'GET':
        # Get all education content
        try:
            education_data = list(education_collection.find().sort('timestamp', -1))
            for item in education_data:
                item['_id'] = str(item['_id'])
            return jsonify({
                'status': 'success',
                'data': education_data
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    elif request.method == 'POST':
        # Add new education content
        try:
            data = request.json
            if not data.get('question') or not data.get('answer'):
                return jsonify({
                    'status': 'error',
                    'message': 'Question and answer are required'
                }), 400
                
            education_collection.insert_one({
                'question': data['question'],
                'answer': data['answer'],
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            return jsonify({
                'status': 'success',
                'message': 'Education content added successfully'
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    elif request.method == 'DELETE':
        # Delete education content
        try:
            data = request.json
            if not data.get('id'):
                return jsonify({
                    'status': 'error',
                    'message': 'ID is required for deletion'
                }), 400
                
            result = education_collection.delete_one({'_id': ObjectId(data['id'])})
            
            if result.deleted_count == 0:
                return jsonify({
                    'status': 'error',
                    'message': 'Education content not found'
                }), 404
                
            return jsonify({
                'status': 'success',
                'message': 'Education content deleted successfully'
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'endpoints': [
            '/health - Check server health',
            '/predict_cibil - Predict CIBIL score impact and recovery',
            '/predict_default - Predict loan default probability',
            '/forecast_scenarios - Generate scenario-based forecasts',
            '/train_model - Train or retrain ML models'
        ]
    })

@app.route('/predict_cibil', methods=['POST'])
def predict_cibil_new():
    try:
        data = request.get_json()
        
        # Extract features from request
        current_cibil = data.get('current_cibil', 0)
        loan_amount = data.get('loan_amount', 0)
        tenure_months = data.get('tenure_months', 0)
        interest_rate = data.get('interest_rate', 0)
        monthly_income = data.get('monthly_income', 0)
        existing_loans = data.get('existing_loans', 0)
        credit_card_limit = data.get('credit_card_limit', 0)
        credit_card_balance = data.get('credit_card_balance', 0)
        age = data.get('age', 30)
        existing_emis = data.get('existing_emis', 0)
        payment_history = data.get('payment_history', 100)  # 0-100 scale, 100 being perfect
        
        # Calculate credit utilization ratio
        credit_utilization = credit_card_balance / credit_card_limit if credit_card_limit > 0 else 0
        
        # Calculate debt-to-income ratio
        new_loan_emi = calculate_emi(loan_amount, interest_rate, tenure_months)
        total_monthly_obligations = existing_emis + new_loan_emi
        debt_to_income = (total_monthly_obligations / monthly_income) if monthly_income > 0 else 0
        
        # Prepare features for the model
        features = [
            current_cibil, 
            loan_amount, 
            tenure_months, 
            interest_rate,
            monthly_income,
            existing_loans,
            credit_utilization * 100,  # Convert to percentage
            debt_to_income * 100,      # Convert to percentage
            age,
            payment_history
        ]
        
        # If model is loaded, make prediction
        if cibil_model and cibil_scaler:
            # Scale features
            features_scaled = cibil_scaler.transform([features])
            
            # Predict immediate impact on CIBIL score
            immediate_impact = cibil_model.predict(features_scaled)[0]
            
            # Calculate recovery timeline
            recovery_timeline = []
            current_score = current_cibil + immediate_impact
            
            # Simulate score recovery over 24 months
            for month in range(1, 25):
                if month % 3 == 0:  # Update every 3 months
                    recovery_features = features.copy()
                    recovery_features[0] = current_score  # Update current CIBIL
                    recovery_features[1] = loan_amount * (1 - (month / tenure_months)) if month < tenure_months else 0
                    
                    # Assume improving payment history over time
                    if payment_history < 100:
                        recovery_features[9] = min(100, payment_history + (month * 0.5))
                    
                    # Predict score for this month
                    recovery_features_scaled = cibil_scaler.transform([recovery_features])
                    month_impact = cibil_model.predict(recovery_features_scaled)[0]
                    
                    current_score = current_score + month_impact
                    
                    # Ensure score stays within realistic bounds (300-900 for CIBIL)
                    current_score = max(300, min(900, current_score))
                    
                    recovery_timeline.append({
                        'month': month,
                        'score': round(current_score, 2)
                    })
            
            return jsonify({
                'status': 'success',
                'current_cibil': current_cibil,
                'immediate_impact': round(immediate_impact, 2),
                'predicted_cibil': round(current_cibil + immediate_impact, 2),
                'recovery_timeline': recovery_timeline,
                'monthly_loan_payment': round(new_loan_emi, 2),
                'debt_to_income_ratio': round(debt_to_income * 100, 2),
                'credit_utilization': round(credit_utilization * 100, 2)
            })
        else:
            # Fallback calculation if model isn't loaded
            return fallback_cibil_prediction(
                current_cibil, loan_amount, tenure_months, 
                interest_rate, monthly_income, existing_emis,
                credit_utilization, debt_to_income
            )
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }), 500

@app.route('/predict_default', methods=['POST'])
def predict_default():
    try:
        data = request.get_json()
        
        # Extract features
        cibil_score = data.get('cibil_score', 0)
        loan_amount = data.get('loan_amount', 0)
        tenure_months = data.get('tenure_months', 0)
        interest_rate = data.get('interest_rate', 0)
        monthly_income = data.get('monthly_income', 0)
        existing_loans = data.get('existing_loans', 0)
        employment_years = data.get('employment_years', 0)
        age = data.get('age', 30)
        
        # Calculate EMI and DTI
        emi = calculate_emi(loan_amount, interest_rate, tenure_months)
        debt_to_income = (emi / monthly_income) if monthly_income > 0 else 0
        
        features = [
            cibil_score,
            loan_amount,
            tenure_months,
            interest_rate,
            monthly_income,
            existing_loans,
            debt_to_income * 100,
            employment_years,
            age
        ]
        
        if default_model and default_scaler:
            # Scale features
            features_scaled = default_scaler.transform([features])
            
            # Predict default probability
            default_prob = default_model.predict_proba(features_scaled)[0][1]  # Probability of class 1 (default)
            
            # Get feature importances
            if hasattr(default_model, 'feature_importances_'):
                importances = default_model.feature_importances_
                feature_names = ['CIBIL Score', 'Loan Amount', 'Tenure', 'Interest Rate', 
                                'Monthly Income', 'Existing Loans', 'DTI Ratio', 
                                'Employment Years', 'Age']
                
                importance_dict = dict(zip(feature_names, importances.tolist()))
            else:
                importance_dict = {}
            
            return jsonify({
                'status': 'success',
                'default_probability': round(default_prob * 100, 2),
                'risk_category': categorize_risk(default_prob),
                'monthly_payment': round(emi, 2),
                'debt_to_income': round(debt_to_income * 100, 2),
                'feature_importance': importance_dict
            })
        else:
            # Fallback if model not loaded
            return fallback_default_prediction(
                cibil_score, loan_amount, monthly_income, debt_to_income
            )
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Default prediction error: {str(e)}'
        }), 500

@app.route('/forecast_scenarios', methods=['POST'])
def forecast_scenarios():
    try:
        data = request.get_json()
        
        # Base data
        current_cibil = data.get('current_cibil', 0)
        monthly_income = data.get('monthly_income', 0)
        existing_emis = data.get('existing_emis', 0)
        credit_card_limit = data.get('credit_card_limit', 0)
        credit_card_balance = data.get('credit_card_balance', 0)
        
        # Scenarios vary by loan amount, tenure, and interest rate
        scenarios = []
        
        # Scenario inputs
        loan_amounts = data.get('loan_amounts', [100000, 300000, 500000])
        tenures = data.get('tenures', [12, 36, 60])
        interest_rates = data.get('interest_rates', [8.0, 10.0, 12.0])
        
        # Generate all combinations
        for amount in loan_amounts:
            for tenure in tenures:
                for rate in interest_rates:
                    # Calculate EMI
                    emi = calculate_emi(amount, rate, tenure)
                    
                    # Calculate debt-to-income
                    dti = ((existing_emis + emi) / monthly_income) if monthly_income > 0 else 0
                    
                    # Prepare features for CIBIL prediction
                    features = [
                        current_cibil,
                        amount,
                        tenure,
                        rate,
                        monthly_income,
                        1,  # existing loans
                        (credit_card_balance / credit_card_limit * 100) if credit_card_limit > 0 else 0,
                        dti * 100,
                        30,  # default age
                        100  # perfect payment history
                    ]
                    
                    # Predict CIBIL impact
                    if cibil_model and cibil_scaler:
                        features_scaled = cibil_scaler.transform([features])
                        impact = cibil_model.predict(features_scaled)[0]
                        new_cibil = current_cibil + impact
                    else:
                        # Simple fallback
                        impact = -(dti * 50)  # Higher DTI = more negative impact
                        new_cibil = current_cibil + impact
                    
                    # Predict default probability
                    default_prob = 0
                    if default_model and default_scaler:
                        default_features = [
                            current_cibil,
                            amount,
                            tenure,
                            rate,
                            monthly_income,
                            1,  # existing loans
                            dti * 100,
                            5,  # default employment years
                            30  # default age
                        ]
                        default_features_scaled = default_scaler.transform([default_features])
                        default_prob = default_model.predict_proba(default_features_scaled)[0][1]
                    
                    scenarios.append({
                        'loan_amount': amount,
                        'tenure_months': tenure,
                        'interest_rate': rate,
                        'monthly_payment': round(emi, 2),
                        'cibil_impact': round(impact, 2),
                        'predicted_cibil': round(new_cibil, 2),
                        'default_probability': round(default_prob * 100, 2),
                        'risk_category': categorize_risk(default_prob),
                        'debt_to_income': round(dti * 100, 2),
                        'affordability_index': calculate_affordability(dti)
                    })
        
        # Sort scenarios by CIBIL impact (least negative to most)
        scenarios.sort(key=lambda x: x['cibil_impact'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'current_cibil': current_cibil,
            'scenarios': scenarios,
            'recommendation': get_recommendation(scenarios)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Forecasting error: {str(e)}'
        }), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        
        # Check for training flag
        should_train = data.get('train', False)
        
        if not should_train:
            return jsonify({
                'status': 'skipped',
                'message': 'Training not requested'
            })
        
        # Load dataset (should be provided or loaded from CSV)
        dataset_path = os.path.join(model_dir, 'cibil_data.csv')
        if not os.path.exists(dataset_path):
            return jsonify({
                'status': 'error',
                'message': 'Training dataset not found'
            }), 404
        
        # Load and prepare data
        df = pd.read_csv(dataset_path)
        
        # Train CIBIL prediction model
        X_cibil = df[['current_cibil', 'loan_amount', 'tenure_months', 'interest_rate', 
                      'monthly_income', 'existing_loans', 'credit_utilization', 
                      'debt_to_income', 'age', 'payment_history']]
        y_cibil = df['cibil_impact']
        
        # Scale features
        scaler = StandardScaler()
        X_cibil_scaled = scaler.fit_transform(X_cibil)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_cibil_scaled, y_cibil)
        
        # Save models
        joblib.dump(model, os.path.join(model_dir, 'cibil_model.pkl'))
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        # Load the new models
        global cibil_model, cibil_scaler
        cibil_model = model
        cibil_scaler = scaler
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained and saved successfully',
            'model_info': {
                'type': 'RandomForestRegressor',
                'features': X_cibil.columns.tolist(),
                'training_samples': len(X_cibil)
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Training error: {str(e)}'
        }), 500

# Utility Functions
def calculate_emi(principal, rate, tenure):
    """Calculate EMI (Equated Monthly Installment)"""
    monthly_rate = rate / (12 * 100)
    emi = (principal * monthly_rate * ((1 + monthly_rate) ** tenure)) / (((1 + monthly_rate) ** tenure) - 1)
    return emi

def categorize_risk(default_probability):
    """Categorize risk based on default probability"""
    if default_probability < 0.1:
        return "Very Low Risk"
    elif default_probability < 0.2:
        return "Low Risk"
    elif default_probability < 0.4:
        return "Moderate Risk"
    elif default_probability < 0.6:
        return "High Risk"
    else:
        return "Very High Risk"

def calculate_affordability(dti_ratio):
    """Calculate affordability index from DTI ratio"""
    if dti_ratio < 0.3:
        return "Highly Affordable"
    elif dti_ratio < 0.4:
        return "Affordable"
    elif dti_ratio < 0.5:
        return "Moderately Affordable"
    elif dti_ratio < 0.6:
        return "Stretching Budget"
    else:
        return "Not Affordable"

def get_recommendation(scenarios):
    """Generate recommendation based on scenarios"""
    if not scenarios:
        return "No viable loan scenarios found"
    
    # Filter to affordable scenarios
    affordable = [s for s in scenarios if s['debt_to_income'] < 50]
    
    if not affordable:
        return "Consider reducing loan amount or increasing tenure for better affordability"
    
    # Find best scenario (highest CIBIL with lowest default prob)
    best = max(affordable, key=lambda x: x['predicted_cibil'] - (x['default_probability'] * 5))
    
    return {
        'message': "Recommended loan scenario based on CIBIL impact and risk",
        'recommended_scenario': {
            'loan_amount': best['loan_amount'],
            'tenure_months': best['tenure_months'],
            'interest_rate': best['interest_rate'],
            'monthly_payment': best['monthly_payment'],
            'cibil_impact': best['cibil_impact'],
            'risk_category': best['risk_category']
        }
    }

def fallback_cibil_prediction(current_cibil, loan_amount, tenure, 
                             interest_rate, monthly_income, existing_emis,
                             credit_utilization, debt_to_income):
    """Fallback prediction logic when model isn't available"""
    # Calculate base impact based on DTI
    base_impact = -10 if debt_to_income > 0.5 else -5
    
    # Adjust based on loan amount relative to income
    loan_to_income = loan_amount / (monthly_income * 12) if monthly_income > 0 else 0
    amount_factor = -15 if loan_to_income > 3 else -5 if loan_to_income > 1 else 0
    
    # Adjust based on credit utilization
    utilization_factor = -10 if credit_utilization > 80 else -5 if credit_utilization > 50 else 0
    
    # Total immediate impact
    impact = base_impact + amount_factor + utilization_factor
    
    # Generate simple recovery timeline
    recovery_timeline = []
    predicted_cibil = current_cibil + impact
    
    for month in range(3, 25, 3):
        recovery_rate = 2  # Points recovered per quarter with on-time payments
        predicted_cibil += recovery_rate
        predicted_cibil = min(current_cibil, predicted_cibil)  # Don't exceed original score
        
        recovery_timeline.append({
            'month': month,
            'score': round(predicted_cibil, 2)
        })
    
    return jsonify({
        'status': 'success',
        'note': 'Using fallback prediction (model not loaded)',
        'current_cibil': current_cibil,
        'immediate_impact': round(impact, 2),
        'predicted_cibil': round(current_cibil + impact, 2),
        'recovery_timeline': recovery_timeline,
        'monthly_loan_payment': round(calculate_emi(loan_amount, interest_rate, tenure), 2),
        'debt_to_income_ratio': round(debt_to_income * 100, 2),
        'credit_utilization': round(credit_utilization * 100, 2)
    })

def fallback_default_prediction(cibil_score, loan_amount, monthly_income, debt_to_income):
    """Fallback default prediction when model isn't available"""
    # Base probability based on CIBIL score
    if cibil_score >= 750:
        base_prob = 0.05
    elif cibil_score >= 650:
        base_prob = 0.15
    elif cibil_score >= 550:
        base_prob = 0.30
    else:
        base_prob = 0.50
    
    # Adjust for DTI
    dti_factor = debt_to_income * 0.5  # Higher DTI = higher risk
    
    # Adjust for loan amount to income ratio
    annual_income = monthly_income * 12
    loan_to_income = loan_amount / annual_income if annual_income > 0 else 0
    amount_factor = min(0.3, loan_to_income * 0.1)
    
    # Calculate final probability
    default_prob = base_prob + dti_factor + amount_factor
    default_prob = min(0.95, max(0.01, default_prob))  # Keep between 1% and 95%
    
    return jsonify({
        'status': 'success',
        'note': 'Using fallback prediction (model not loaded)',
        'default_probability': round(default_prob * 100, 2),
        'risk_category': categorize_risk(default_prob),
        'debt_to_income': round(debt_to_income * 100, 2)
    })

if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Initialize model on startup
    try:
        model.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Training new model...")
        model.train()
    
    # Print server info
    print("=" * 50)
    print("Flask server starting...")
    print("Available endpoints:")
    print("  - POST /predict_cibil")
    print("  - GET /train")
    print("  - GET /initialize")
    print("  - GET /health")
    print("  - GET /users")
    print("  - GET /cibil_data")
    print("  - GET /import_csv")
    print("  - GET/POST/DELETE /education")
    print("  - POST /predict_cibil")
    print("  - POST /predict_default")
    print("  - POST /forecast_scenarios")
    print("  - POST /train_model")
    print("=" * 50)
    
    # Run the app on all network interfaces, port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)