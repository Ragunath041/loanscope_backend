from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['loanscope']
cibil_collection = db['cibil_data']

# Read CSV file
df = pd.read_csv('cibil_data.csv')

# Convert DataFrame to list of dictionaries
records = df.to_dict('records')

# Drop existing collection
cibil_collection.drop()

# Insert records
result = cibil_collection.insert_many(records)

print(f"Successfully imported {len(result.inserted_ids)} records")
