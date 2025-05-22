import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pymongo import MongoClient

class CibilScoreModel:
    """
    Class to handle CIBIL score prediction and loan default prediction
    Includes methods for data preparation, model training, evaluation, and prediction
    """
    
    def __init__(self, model_dir='model'):
        """Initialize the model with the directory to save/load models"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # CIBIL prediction model
        self.cibil_model = None
        self.cibil_scaler = None
        
        # Default prediction model
        self.default_model = None
        self.default_scaler = None
        
        # Connect to MongoDB
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['loanscope']
        self.cibil_collection = self.db['cibil_data']
        
    def prepare_data(self, csv_path='cibil_data.csv'):
        """Prepare data for model training"""
        # Load the data
        data = pd.read_csv(csv_path)
        
        # Data preprocessing for CIBIL prediction model
        # Extract features and target for CIBIL impact prediction
        cibil_features = data[['CIBIL', 'Sanctioned_Amount', 'Loan_Tenure', 
                              'Interest_Rate', 'Monthly_Income', 'Previous_Loans', 
                              'Credit_Utilization', 'Debt_to_Income_Ratio', 
                              'DOB']].copy()
        
        # Convert DOB to age
        cibil_features['DOB'] = pd.to_datetime(cibil_features['DOB'], format='%d-%m-%Y', errors='coerce')
        current_year = pd.Timestamp.now().year
        cibil_features['Age'] = current_year - cibil_features['DOB'].dt.year
        cibil_features.drop('DOB', axis=1, inplace=True)
        
        # Create payment history score (derived from Loan_Repayment_History)
        payment_history = pd.Series(np.where(data['Loan_Repayment_History'] == 'Good', 100,
                                  np.where(data['Loan_Repayment_History'] == 'Average', 75, 50)))
        cibil_features['Payment_History'] = payment_history
            
            # Handle missing values
        cibil_features.fillna({
            'CIBIL': data['CIBIL'].median(),
            'Sanctioned_Amount': data['Sanctioned_Amount'].median(),
            'Loan_Tenure': data['Loan_Tenure'].median(),
            'Interest_Rate': data['Interest_Rate'].median(),
            'Monthly_Income': data['Monthly_Income'].median(),
            'Previous_Loans': data['Previous_Loans'].median(),
            'Credit_Utilization': 0,
            'Debt_to_Income_Ratio': 0.1,
            'Age': 30,
            'Payment_History': 75
        }, inplace=True)
        
        # Create synthetic target for CIBIL impact
        # This calculates how much the CIBIL score might change based on the data
        # In a real scenario, you would have historical data showing actual changes
        
        # Higher debt-to-income ratio reduces score
        dti_impact = -10 * data['Debt_to_Income_Ratio']
        
        # Late payments reduce score
        late_payment_impact = np.where(data['Late_Payment'] == 'YES', -15, 0)
        
        # Good repayment history improves score
        history_impact = np.where(data['Loan_Repayment_History'] == 'Good', 5,
                                np.where(data['Loan_Repayment_History'] == 'Average', 0, -5))
        
        # Create the target variable as the combined impact
        cibil_impact = dti_impact + late_payment_impact + history_impact
        
        # Add some random variation to make the model more realistic
        np.random.seed(42)
        random_variation = np.random.normal(0, 3, size=len(cibil_impact))
        cibil_impact += random_variation
        
        # Rename columns to match the expected format in Flask app
        cibil_features.rename(columns={
            'CIBIL': 'current_cibil',
            'Sanctioned_Amount': 'loan_amount',
            'Loan_Tenure': 'tenure_months',
            'Interest_Rate': 'interest_rate',
            'Monthly_Income': 'monthly_income',
            'Previous_Loans': 'existing_loans',
            'Credit_Utilization': 'credit_utilization',
            'Debt_to_Income_Ratio': 'debt_to_income',
            'Payment_History': 'payment_history'
        }, inplace=True)
        
        # Data preprocessing for default prediction model
        # Create a binary target for default (1 = default, 0 = no default)
        # Here we're creating synthetic data based on domain knowledge
        default_target = np.zeros(len(data))
        
        # Higher probability of default if:
        # - CIBIL score is low
        # - Debt-to-income ratio is high
        # - Late payments
        # - Defaults history
        
        low_cibil_mask = data['CIBIL'] < 600
        high_dti_mask = data['Debt_to_Income_Ratio'] > 0.4
        late_payment_mask = data['Late_Payment'] == 'YES'
        defaults_mask = data['Defaults'] > 0
        
        # Assign default=1 based on these factors
        default_target[low_cibil_mask & high_dti_mask] = 1
        default_target[late_payment_mask & defaults_mask] = 1
        default_target[low_cibil_mask & late_payment_mask] = 1
        default_target[high_dti_mask & defaults_mask] = 1
        
        # Add some randomness to make it realistic
        np.random.seed(43)
        random_defaults = np.random.binomial(1, 0.15, size=len(default_target))
        default_target = np.where(random_defaults == 1, default_target, random_defaults)
        
        # Create features for default prediction
        default_features = data[['CIBIL', 'Sanctioned_Amount', 'Loan_Tenure', 
                                'Interest_Rate', 'Monthly_Income', 'Previous_Loans', 
                                'Debt_to_Income_Ratio']].copy()
        
        # Add employment years (derived from Employment_Type)
        employment_years = pd.Series(np.where(data['Employment_Type'] == 'Salaried', 5,
                                  np.where(data['Employment_Type'] == 'Self-Employed', 3, 1)))
        default_features['Employment_Years'] = employment_years
        
        # Add age
        default_features['Age'] = cibil_features['Age']
        
        # Handle missing values
        default_features.fillna({
            'CIBIL': data['CIBIL'].median(),
            'Sanctioned_Amount': data['Sanctioned_Amount'].median(),
            'Loan_Tenure': data['Loan_Tenure'].median(),
            'Interest_Rate': data['Interest_Rate'].median(),
            'Monthly_Income': data['Monthly_Income'].median(),
            'Previous_Loans': data['Previous_Loans'].median(),
            'Debt_to_Income_Ratio': 0.1,
            'Employment_Years': 3,
            'Age': 30
        }, inplace=True)
        
        # Rename columns to match the expected format in Flask app
        default_features.rename(columns={
            'CIBIL': 'cibil_score',
            'Sanctioned_Amount': 'loan_amount',
            'Loan_Tenure': 'tenure_months',
            'Interest_Rate': 'interest_rate',
            'Monthly_Income': 'monthly_income',
            'Previous_Loans': 'existing_loans',
            'Debt_to_Income_Ratio': 'debt_to_income'
        }, inplace=True)
        
        # Multiply debt_to_income by 100 to match the expected format
        default_features['debt_to_income'] = default_features['debt_to_income'] * 100
        cibil_features['debt_to_income'] = cibil_features['debt_to_income'] * 100
        
        return cibil_features, cibil_impact, default_features, default_target
    
    def train(self):
        """Train the CIBIL score and default prediction models"""
        # Prepare data
        cibil_features, cibil_impact, default_features, default_target = self.prepare_data()
        
        # Split data for CIBIL prediction
        X_cibil_train, X_cibil_test, y_cibil_train, y_cibil_test = train_test_split(
            cibil_features, cibil_impact, test_size=0.2, random_state=42
        )
        
        # Scale features for CIBIL prediction
        self.cibil_scaler = StandardScaler()
        X_cibil_train_scaled = self.cibil_scaler.fit_transform(X_cibil_train)
        X_cibil_test_scaled = self.cibil_scaler.transform(X_cibil_test)
        
        # Train CIBIL prediction model
        print("Training CIBIL prediction model...")
        self.cibil_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.cibil_model.fit(X_cibil_train_scaled, y_cibil_train)
        
        # Evaluate CIBIL prediction model
        y_cibil_pred = self.cibil_model.predict(X_cibil_test_scaled)
        cibil_mse = mean_squared_error(y_cibil_test, y_cibil_pred)
        cibil_rmse = np.sqrt(cibil_mse)
        
        print(f"CIBIL prediction model - RMSE: {cibil_rmse:.2f}")
        
        # Split data for default prediction
        X_default_train, X_default_test, y_default_train, y_default_test = train_test_split(
            default_features, default_target, test_size=0.2, random_state=42, stratify=default_target
        )
        
        # Scale features for default prediction
        self.default_scaler = StandardScaler()
        X_default_train_scaled = self.default_scaler.fit_transform(X_default_train)
        X_default_test_scaled = self.default_scaler.transform(X_default_test)
        
        # Train default prediction model
        print("Training default prediction model...")
        self.default_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.default_model.fit(X_default_train_scaled, y_default_train)
        
        # Evaluate default prediction model
        y_default_pred = self.default_model.predict(X_default_test_scaled)
        y_default_prob = self.default_model.predict_proba(X_default_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_default_test, y_default_pred)
        precision = precision_score(y_default_test, y_default_pred)
        recall = recall_score(y_default_test, y_default_pred)
        f1 = f1_score(y_default_test, y_default_pred)
        auc = roc_auc_score(y_default_test, y_default_prob)
        
        print(f"Default prediction model - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")
        
        # Save models
        self.save_model()
            
        return {
            'cibil_model': {
                'rmse': cibil_rmse,
                'feature_importance': dict(zip(cibil_features.columns, self.cibil_model.feature_importances_))
            },
            'default_model': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'feature_importance': dict(zip(default_features.columns, self.default_model.feature_importances_))
            }
        }
    
    def save_model(self):
        """Save models to disk"""
        # Save CIBIL prediction model and scaler
        joblib.dump(self.cibil_model, os.path.join(self.model_dir, 'cibil_model.pkl'))
        joblib.dump(self.cibil_scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        
        # Save default prediction model and scaler
        joblib.dump(self.default_model, os.path.join(self.model_dir, 'default_model.pkl'))
        joblib.dump(self.default_scaler, os.path.join(self.model_dir, 'default_scaler.pkl'))
        
        print("Models saved successfully")
    
    def load_model(self):
        """Load models from disk"""
        # Load CIBIL prediction model and scaler
        self.cibil_model = joblib.load(os.path.join(self.model_dir, 'cibil_model.pkl'))
        self.cibil_scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        
        # Load default prediction model and scaler
        self.default_model = joblib.load(os.path.join(self.model_dir, 'default_model.pkl'))
        self.default_scaler = joblib.load(os.path.join(self.model_dir, 'default_scaler.pkl'))
        
        print("Models loaded successfully")
    
    def predict_cibil_impact(self, features):
        """Predict CIBIL score impact based on features"""
        # Scale features
        features_scaled = self.cibil_scaler.transform([features])
            
            # Make prediction
        impact = self.cibil_model.predict(features_scaled)[0]
        
        return impact
    
    def predict_default_probability(self, features):
        """Predict default probability based on features"""
        # Scale features
        features_scaled = self.default_scaler.transform([features])
        
        # Make prediction
        default_prob = self.default_model.predict_proba(features_scaled)[0][1]
        
        return default_prob
    
    def create_training_data(self, output_file='cibil_data.csv', num_samples=300):
        """Create synthetic training data for model development"""
        np.random.seed(42)
        
        # Generate random data
        data = {
            'Name': [f'Person_{i}' for i in range(num_samples)],
            'PAN': [f'PAN{i:05d}' for i in range(num_samples)],
            'CIBIL': np.random.randint(300, 900, num_samples),
            'DOB': [f'{np.random.randint(1, 29):02d}-{np.random.randint(1, 13):02d}-{np.random.randint(1960, 2000)}' for _ in range(num_samples)],
            'Loan_Type': np.random.choice(['Personal Loan', 'Home Loan', 'Car Loan', 'Education Loan'], num_samples),
            'Sanctioned_Amount': np.random.randint(100000, 5000000, num_samples),
            'Current_Amount': np.random.randint(50000, 5000000, num_samples),
            'Credit_Card': np.random.choice(['YES', 'NO'], num_samples),
            'Late_Payment': np.random.choice(['YES', 'NO'], num_samples, p=[0.3, 0.7]),
            'Loan_Tenure': np.random.randint(1, 30, num_samples),
            'Interest_Rate': np.random.uniform(7.0, 18.0, num_samples),
            'Monthly_Income': np.random.randint(20000, 200000, num_samples),
            'Monthly_EMI': np.random.randint(1000, 50000, num_samples),
            'Previous_Loans': np.random.randint(0, 6, num_samples),
            'Defaults': np.random.randint(0, 3, num_samples),
            'Credit_Cards_Count': np.random.randint(0, 6, num_samples),
            'Credit_Utilization': np.random.randint(0, 100, num_samples),
            'Loan_Repayment_History': np.random.choice(['Good', 'Average', 'Poor'], num_samples, p=[0.6, 0.3, 0.1]),
            'Other_Debts': np.random.choice(['YES', 'NO'], num_samples),
            'Employment_Type': np.random.choice(['Salaried', 'Self-Employed', 'Freelancer'], num_samples),
            'Existing_EMIs': np.random.randint(0, 4, num_samples),
            'Savings_Balance': np.random.randint(10000, 500000, num_samples),
            'Total_Annual_Income': []
        }
        
        # Calculate annual income from monthly
        for income in data['Monthly_Income']:
            annual = income * 12
            # Add some variation
            data['Total_Annual_Income'].append(annual + np.random.randint(-50000, 50000))
        
        # Calculate debt-to-income ratio
        data['Debt_to_Income_Ratio'] = []
        for emi, income in zip(data['Monthly_EMI'], data['Monthly_Income']):
            if income > 0:
                dti = emi / income
                # Cap at a reasonable value
                dti = min(dti, 0.8)
            else:
                dti = 0
            data['Debt_to_Income_Ratio'].append(round(dti, 3))
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        print(f"Created training data with {num_samples} samples and saved to {output_file}")
        
        return df

def main():
    try:
        # Train and save the model
        model = CibilScoreModel()
        model.train()
        
        print("Model training completed successfully!")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()