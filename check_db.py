import requests
import pymongo
import time

def check_mongodb():
    """Check if MongoDB is running and accessible"""
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        server_info = client.server_info()
        print("✅ MongoDB is running")
        print(f"   Version: {server_info.get('version')}")
        
        db = client['loanscope']
        collections = db.list_collection_names()
        print(f"   Database: loanscope")
        print(f"   Collections: {len(collections)}")
        for collection in collections:
            count = db[collection].count_documents({})
            print(f"     - {collection}: {count} documents")
        
        return True
    except pymongo.errors.ServerSelectionTimeoutError:
        print("❌ MongoDB is not running or not accessible")
        print("   Please start MongoDB server")
        return False
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {str(e)}")
        return False

def check_flask():
    """Check if Flask API is running and accessible"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print("✅ Flask API is running")
            print(f"   Status: {data.get('status')}")
            print(f"   Message: {data.get('message')}")
            print(f"   MongoDB connection: {data.get('mongo_connection')}")
            return True
        else:
            print(f"❌ Flask API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Flask API is not running or not accessible")
        print("   Please start Flask with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Flask API: {str(e)}")
        return False

if __name__ == "__main__":
    print("Checking database and API services...")
    print("-" * 50)
    
    mongo_status = check_mongodb()
    print("-" * 50)
    
    flask_status = check_flask()
    print("-" * 50)
    
    if mongo_status and flask_status:
        print("✅ All services are running properly!")
    else:
        print("⚠️ Some services are not running properly.")
        
        if not mongo_status:
            print("\nTo start MongoDB on Windows:")
            print("1. Run Command Prompt as Administrator")
            print("2. Type: net start MongoDB")
            print("   Or manually start MongoDB from the Services app")
        
        if not flask_status:
            print("\nTo start Flask API:")
            print("1. Open a terminal in the backend directory")
            print("2. Run: python app.py")
    
    print("\nPress Enter to exit...")
    input() 