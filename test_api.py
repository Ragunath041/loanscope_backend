import requests
import json
import datetime

def test_register():
    """Test the registration API endpoint"""
    # Use a timestamp to create a unique email
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    email = f"test_user_{timestamp}@example.com"
    
    url = "http://localhost:5000/register"
    data = {
        "email": email,
        "password": "password123",
        "name": f"Test User {timestamp}",
        "pan_number": f"TEST{timestamp[-6:]}Z"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending registration request to: {url}")
        print(f"Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, json=data, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print(f"Successfully registered with email: {email}")
            # Return both the response and the credentials
            return {
                "response": response.json(),
                "email": email,
                "password": "password123"
            }
        else:
            return {
                "response": response.json(),
                "email": None,
                "password": None
            }
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def test_login(email="test_user@example.com", password="password123"):
    """Test the login API endpoint"""
    url = "http://localhost:5000/login"
    data = {
        "email": email,
        "password": password
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending login request to: {url}")
        print(f"Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, json=data, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.json()
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def test_health():
    """Test the health check endpoint"""
    url = "http://localhost:5000/health"
    
    try:
        print(f"Sending health check request to: {url}")
        
        response = requests.get(url)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.json()
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    
if __name__ == "__main__":
    print("=" * 50)
    print("Testing API Endpoints")
    print("=" * 50)
    
    # Test health check
    print("\nHealth Check:")
    test_health()
    
    # Test registration
    print("\nRegistration Test:")
    reg_result = test_register()
    
    # Test login
    print("\nLogin Test:")
    if reg_result and reg_result["response"].get('status') == 'success':
        email = reg_result["email"]
        password = reg_result["password"]
        print(f"Using new credentials - Email: {email}")
        test_login(email, password)
    else:
        print("Registration failed, trying to login with existing credentials")
        test_login()
    
    print("\nTesting completed!") 