from pymongo import MongoClient
import os
from dotenv import load_dotenv
import datetime

# Load environment variables
load_dotenv()

# Get MongoDB connection details
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB = os.getenv('MONGODB_DB')

if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable is not set")

# Connect to MongoDB
client = MongoClient(MONGODB_URI)

# Create database and collections
print(f"\nConnecting to MongoDB Atlas...")
db = client[MONGODB_DB]

# Insert test data to verify database creation
print("\nCreating test data...")

# Test data for users collection
test_user = {
    "email": "test@example.com",
    "pan_number": "ABCDE1234F",
    "name": "Test User",
    "created_at": datetime.datetime.utcnow()
}

# Test data for cibil_data collection
test_cibil = {
    "pan_number": "ABCDE1234F",
    "cibil_score": 750,
    "created_at": datetime.datetime.utcnow()
}

# Insert test data
print("\nInserting test data...")
try:
    # Insert into users collection
    user_result = db.users.insert_one(test_user)
    print(f"Test user inserted with ID: {user_result.inserted_id}")
    
    # Insert into cibil_data collection
    cibil_result = db.cibil_data.insert_one(test_cibil)
    print(f"Test CIBIL data inserted with ID: {cibil_result.inserted_id}")
    
    # Verify collections exist
    print("\nCollections in database:")
    for collection in db.list_collection_names():
        print(f"- {collection}")
        
    print("\nDatabase setup and test data insertion completed successfully!")
    
    # Verify test data exists
    print("\nVerifying test data...")
    user = db.users.find_one({"email": "test@example.com"})
    cibil = db.cibil_data.find_one({"pan_number": "ABCDE1234F"})
    
    if user:
        print("\nTest user data:")
        print(f"Email: {user['email']}")
        print(f"PAN Number: {user['pan_number']}")
    
    if cibil:
        print("\nTest CIBIL data:")
        print(f"CIBIL Score: {cibil['cibil_score']}")
        print(f"PAN Number: {cibil['pan_number']}")
        
except Exception as e:
    print(f"\nError: {str(e)}")
    print("\nDatabase setup failed. Please check your MongoDB Atlas credentials and network access.")
