import pandas as pd
from pymongo import MongoClient
import os
import datetime
import hashlib
import uuid

# Helper function for hashing passwords
def hash_password(password, salt=None):
    """Hash a password for storing."""
    if salt is None:
        salt = uuid.uuid4().hex
    
    hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
    return {'salt': salt, 'hashed': hashed_password}

# Connect to MongoDB
print("Connecting to MongoDB...")
client = MongoClient('mongodb://localhost:27017/')
db = client['loanscope']

# Create collections if they don't exist
collections = ['users', 'cibil_data', 'education', 'loan_applications', 'predictions', 'model_metadata']
existing_collections = db.list_collection_names()

for collection in collections:
    if collection not in existing_collections:
        print(f"Creating collection: {collection}")
        db.create_collection(collection)
    else:
        print(f"Collection {collection} already exists")

# Add sample CIBIL data if collection is empty
if db['cibil_data'].count_documents({}) == 0:
    print("Adding sample CIBIL data...")
    sample_cibil_data = [
        {
            'PAN': 'KOSDE3421L',
            'name': 'Kondalrao',
            'CIBIL': 720,
            'dob': '24-05-2001',
            'loanType': 'Personal Loan',
            'sanctionedAmount': 300000,
            'currentAmount': 350000,
            'creditCard': 'NO',
            'latePayment': 'YES',
            'loanTenure': 9,
            'interestRate': 3.5,
            'monthlyIncome': 25000,
            'monthlyEMI': 30000,
            'previousLoans': 1,
            'defaults': 4,
            'creditCardsCount': 0,
            'creditUtilization': 0,
            'loanRepaymentHistory': 'Good',
            'otherDebts': 'YES',
            'employmentType': 'Freelancer',
            'existingEMIs': 1,
            'savingsBalance': 60000,
            'totalAnnualIncome': 700000,
            'debtToIncomeRatio': 0.004
        },
        {
            'PAN': 'KWFBS3421S',
            'name': 'Kowsic Anand S',
            'CIBIL': 690,
            'dob': '15-02-2002',
            'loanType': 'Home Loan',
            'sanctionedAmount': 1500000,
            'currentAmount': 1502100,
            'creditCard': 'NO',
            'latePayment': 'YES',
            'loanTenure': 20,
            'interestRate': 8.7,
            'monthlyIncome': 45000,
            'monthlyEMI': 75000,
            'previousLoans': 1,
            'defaults': 4,
            'creditCardsCount': 0,
            'creditUtilization': 0,
            'loanRepaymentHistory': 'Good',
            'otherDebts': 'YES',
            'employmentType': 'Freelancer',
            'existingEMIs': 1,
            'savingsBalance': 60000,
            'totalAnnualIncome': 700000,
            'debtToIncomeRatio': 0.004
        },
        {
            'PAN': 'BMGDE3421L',
            'name': 'Ameer Jafar Y',
            'CIBIL': 720,
            'dob': '24-05-2001',
            'loanType': 'Personal Loan',
            'sanctionedAmount': 300000,
            'currentAmount': 350000,
            'creditCard': 'NO',
            'latePayment': 'YES',
            'loanTenure': 9,
            'interestRate': 3.5,
            'monthlyIncome': 25000,
            'monthlyEMI': 30000,
            'previousLoans': 1,
            'defaults': 4,
            'creditCardsCount': 0,
            'creditUtilization': 0,
            'loanRepaymentHistory': 'Good',
            'otherDebts': 'YES',
            'employmentType': 'Freelancer',
            'existingEMIs': 1,
            'savingsBalance': 60000,
            'totalAnnualIncome': 700000,
            'debtToIncomeRatio': 0.004
        },
        {
            'PAN': 'PRSSEA3421L',
            'name': 'Nandhini V',
            'CIBIL': 720,
            'dob': '24-05-2001',
            'loanType': 'Personal Loan',
            'sanctionedAmount': 300000,
            'currentAmount': 350000,
            'creditCard': 'NO',
            'latePayment': 'YES',
            'loanTenure': 9,
            'interestRate': 3.5,
            'monthlyIncome': 25000,
            'monthlyEMI': 30000,
            'previousLoans': 1,
            'defaults': 4,
            'creditCardsCount': 0,
            'creditUtilization': 0,
            'loanRepaymentHistory': 'Good',
            'otherDebts': 'YES',
            'employmentType': 'Freelancer',
            'existingEMIs': 1,
            'savingsBalance': 60000,
            'totalAnnualIncome': 700000,
            'debtToIncomeRatio': 0.004
        }
    ]
    
    db['cibil_data'].insert_many(sample_cibil_data)
    print(f"Added {len(sample_cibil_data)} CIBIL records")

# Add sample users if collection is empty
if db['users'].count_documents({}) == 0:
    print("Adding sample users...")
    
    # Create sample users
    sample_users = [
        {
            'email': 'user1@example.com',
            'password': hash_password('password123'),
            'pan_number': 'ABCDE1234F',
            'name': 'User One',
            'created_at': datetime.datetime.utcnow(),
            'last_login': None
        },
        {
            'email': 'user2@example.com',
            'password': hash_password('password123'),
            'pan_number': 'XYZAB5678G',
            'name': 'User Two',
            'created_at': datetime.datetime.utcnow(),
            'last_login': None
        },
        {
            'email': 'kitcbe.25.21bcb041@gmail.com',
            'password': hash_password('Admin@1234'),
            'pan_number': 'ADMIN',
            'name': 'Admin',
            'created_at': datetime.datetime.utcnow(),
            'last_login': None,
            'is_admin': True
        }
    ]
    
    db['users'].insert_many(sample_users)
    print(f"Added {len(sample_users)} user records")

# Add sample education content if collection is empty
if db['education'].count_documents({}) == 0:
    print("Adding sample education content...")
    sample_education = [
        {
            'question': 'What is a CIBIL score?',
            'answer': 'A CIBIL score is a three-digit number between 300 and 900 that represents your creditworthiness. The higher the score, the better your chances of getting a loan approved.',
            'timestamp': datetime.datetime.utcnow() - datetime.timedelta(days=10)
        },
        {
            'question': 'How can I improve my credit score?',
            'answer': 'You can improve your credit score by paying bills on time, keeping credit card balances low, not applying for too much credit, and maintaining a good mix of credit types.',
            'timestamp': datetime.datetime.utcnow() - datetime.timedelta(days=7)
        },
        {
            'question': 'What factors affect my loan eligibility?',
            'answer': 'Loan eligibility is determined by factors such as credit score, income, existing debt, employment stability, and the purpose of the loan.',
            'timestamp': datetime.datetime.utcnow() - datetime.timedelta(days=5)
        }
    ]
    
    db['education'].insert_many(sample_education)
    print(f"Added {len(sample_education)} education records")

print("Database initialization complete!") 