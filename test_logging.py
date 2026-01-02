#!/usr/bin/env python3
"""
Test script to verify database logging is working correctly
Creates a test prediction and checks if it's saved properly
"""

from database_integration import save_prediction_to_db, get_recent_predictions_from_db, is_database_connected, init_database
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_database_logging():
    """Test database logging functionality"""
    print("🔍 Testing Database Logging...")
    
    # Initialize database connection
    print("🔌 Initializing database connection...")
    db_connected = init_database()
    
    # Check connection first
    if not is_database_connected():
        print("❌ Database not connected")
        return False
    
    print("✅ Database connected")
    
    # Get count before test
    before_count = len(get_recent_predictions_from_db(100))
    print(f"📊 Predictions before test: {before_count}")
    
    # Create test prediction data
    test_patient_data = {
        'patient_name': 'TEST_PATIENT_' + datetime.now().strftime('%H%M%S'),
        'age': 45,
        'sex': 'Male',
        'chest_pain_type': 'Typical Angina',
        'resting_bp': 130,
        'cholesterol': 250,
        'fasting_blood_sugar': '≤ 120 mg/dL',
        'rest_ecg': 'Normal',
        'max_heart_rate': 160,
        'exercise_angina': 'No',
        'st_depression': 1.2,
        'slope': 'Flat',
        'colored_vessels': 1,
        'thalassemia': 'Fixed Defect'
    }
    
    test_prediction_results = {
        'positive_predictions': 2,  # Moderate risk
        'confidence_level': 'Medium',
        'overall_result': '🟡 **MODERATE RISK** (50% model consensus)',
        'recommendation': 'Regular monitoring and lifestyle changes recommended',
        'results': {
            'Logistic Regression': 'No Heart Disease',
            'Random Forest': 'Heart Disease Detected',
            'SVM': 'No Heart Disease',
            'Gradient Boosting': 'Heart Disease Detected'
        },
        'probabilities': {
            'Logistic Regression': {'No Heart Disease': '0.60', 'Heart Disease': '0.40'},
            'Random Forest': {'No Heart Disease': '0.45', 'Heart Disease': '0.55'}
        }
    }
    
    # Test saving prediction
    print("💾 Testing save_prediction_to_db...")
    session_id = save_prediction_to_db(test_patient_data, test_prediction_results)
    
    if session_id:
        print(f"✅ Test prediction saved: {session_id}")
    else:
        print("❌ Failed to save test prediction")
        return False
    
    # Verify it was saved
    print("🔍 Verifying saved prediction...")
    after_count = len(get_recent_predictions_from_db(100))
    print(f"📊 Predictions after test: {after_count}")
    
    if after_count > before_count:
        print("✅ New prediction was successfully logged")
        
        # Get the latest prediction to verify data
        recent = get_recent_predictions_from_db(1)
        if recent and recent[0]['id'] == session_id:
            saved_prediction = recent[0]
            print("✅ Prediction data verified:")
            print(f"   - Patient: {saved_prediction['patient_name']}")
            print(f"   - Risk Level: {saved_prediction['positive_predictions']}")
            print(f"   - Confidence: {saved_prediction['confidence_level']}")
            print(f"   - Result: {saved_prediction['overall_result'][:50]}...")
            return True
        else:
            print("❌ Could not verify saved prediction data")
            return False
    else:
        print("❌ Prediction was not logged to database")
        return False

def test_missing_fields():
    """Test error handling for missing required fields"""
    print("\n🔍 Testing error handling for missing fields...")
    
    # Test with missing patient name
    incomplete_patient_data = {
        'age': 45,
        'sex': 'Male'
        # missing patient_name
    }
    
    incomplete_prediction_results = {
        'positive_predictions': 2,
        'confidence_level': 'Medium'
        # missing overall_result
    }
    
    print("💾 Testing with missing required fields...")
    session_id = save_prediction_to_db(incomplete_patient_data, incomplete_prediction_results)
    
    if session_id is None:
        print("✅ Correctly handled missing required fields")
        return True
    else:
        print("❌ Should have rejected incomplete data")
        return False

if __name__ == "__main__":
    print("🚀 Starting Database Logging Tests...")
    print("=" * 50)
    
    # Test 1: Normal prediction logging
    test1_success = test_database_logging()
    
    # Test 2: Error handling
    test2_success = test_missing_fields()
    
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS:")
    print(f"✅ Database Logging Test: {'PASSED' if test1_success else 'FAILED'}")
    print(f"✅ Error Handling Test: {'PASSED' if test2_success else 'FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! Database logging is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the database configuration.")