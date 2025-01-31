import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import firebase_admin
from firebase_admin import credentials, ml

class CibilScoreModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = 'model/cibil_model.pkl'
        self.scaler_path = 'model/scaler.pkl'
        
    def prepare_data(self):
        try:
            # Read the CSV file
            df = pd.read_csv('cibil_data.csv')
            
            # Select only the relevant numeric columns and target
            relevant_columns = [
                'Sanctioned_Amount',
                'Current_Amount',
                'Loan_Tenure',
                'Monthly_EMI',
                'Previous_Loans',
                'Defaults',
                'Credit_Utilization',
                'Monthly_Income',
                'Late_Payment',
                'CIBIL'  # Target variable
            ]
            
            # Select only relevant columns
            df = df[relevant_columns]
            
            # Convert Late_Payment to numeric (YES/NO to 1/0)
            df['Late_Payment'] = (df['Late_Payment'] == 'YES').astype(int)
            
            # Create feature matrix X and target y
            X = df.drop('CIBIL', axis=1)  # All columns except CIBIL
            y = df['CIBIL']  # Target variable
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Save feature names for prediction
            self.feature_names = X.columns.tolist()
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            raise
    
    def train(self):
        try:
            # Create model directory if it doesn't exist
            os.makedirs('model', exist_ok=True)
            
            # Prepare data
            X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data()
            
            # Initialize and train the model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,  # Let the trees grow fully
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
            # Fit the model
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate and print accuracy metrics
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            print(f"Training Score: {train_score:.4f}")
            print(f"Testing Score: {test_score:.4f}")
            
            # Feature importance
            feature_importance = dict(zip(self.feature_names, 
                                       self.model.feature_importances_))
            print("\nFeature Importance:")
            for feature, importance in sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True):
                print(f"{feature}: {importance:.4f}")
            
            # Save model and scaler
            self.save_model()
            
            return self.model
            
        except Exception as e:
            print(f"Error in train: {str(e)}")
            raise
    
    def save_model(self):
        try:
            # Save the model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save the scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature names
            with open('model/feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
                
            print(f"Model saved to {self.model_path}")
            print(f"Scaler saved to {self.scaler_path}")
            
        except Exception as e:
            print(f"Error in save_model: {str(e)}")
            raise
    
    def load_model(self):
        try:
            # Load the model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load the scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature names
            with open('model/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
                
        except Exception as e:
            print(f"Error in load_model: {str(e)}")
            raise
    
    def predict(self, input_data):
        try:
            if self.model is None:
                self.load_model()
            
            # Create input array with the same features used in training
            input_values = []
            for feature in self.feature_names:
                if feature in input_data:
                    input_values.append(float(input_data[feature]))
                else:
                    # If feature is missing, use a default value
                    print(f"Warning: Missing feature {feature} in input data")
                    input_values.append(0.0)
            
            input_array = np.array([input_values])
            
            # Scale input
            input_scaled = self.scaler.transform(input_array)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            # CIBIL scores are typically between 300 and 900
            prediction = max(300, min(900, int(round(prediction))))
            
            return prediction
            
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            raise

# Initialize Firebase
def initialize_firebase():
    try:
        cred = credentials.Certificate('./credentials/loanscope-9f38b-firebase-adminsdk-fbsvc-3b3f380bbc.json')
        if not firebase_admin._apps:  # Check if already initialized
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'loanscope-9f38b.appspot.com'  # Add your Firebase storage bucket
            })
    except Exception as e:
        print(f"Error initializing Firebase: {str(e)}")
        raise

# Deploy model to Firebase ML
def deploy_to_firebase(model):
    try:
        import tensorflow as tf
        from firebase_admin import storage
        
        # Get bucket
        bucket = storage.bucket()
        
        # Convert the model to TensorFlow format
        model_path = 'model/model.tflite'
        
        # Create a TF SavedModel format with proper weights and bias
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(len(model.feature_names),))
        ])
        
        # Initialize the weights properly
        initial_weights = [
            model.model.feature_importances_.reshape(-1, 1),  # weights
            np.array([0.0])  # bias
        ]
        tf_model.set_weights(initial_weights)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        
        # Upload to Firebase Storage
        blob = bucket.blob('models/cibil_model.tflite')
        blob.upload_from_filename(model_path)
        
        # Get the public URL
        model_url = blob.public_url
        
        # Create Firebase ML model
        firebase_model = ml.Model(
            display_name='cibil_score_predictor',
            tags=['cibil', 'credit_score'],
            model_format=ml.TFLiteFormat(
                model_source=ml.TFLiteGCSModelSource.from_uri(model_url)
            )
        )
        
        # Upload model to Firebase ML
        firebase_model = ml.create_model(firebase_model)
        print(f"Model successfully deployed to Firebase ML with name: {firebase_model.display_name}")
        return firebase_model
        
    except Exception as e:
        print(f"Error deploying to Firebase: {str(e)}")
        raise

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