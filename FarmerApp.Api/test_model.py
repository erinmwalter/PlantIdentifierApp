# test_model.py - Put this in FarmerApp.Api/ folder

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from services.prediction_service import PredictionService
from PIL import Image
import io

def test_prediction(image_path):
    """
    Test the model with a single image
    
    Args:
        image_path: Path to the image file to test
    """
    print("="*60)
    print("PLANT DISEASE MODEL TESTER")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return
    
    print(f"\n📁 Loading image: {image_path}")
    
    try:
        # Initialize the prediction service (loads model)
        print("\nLoading model and class labels...")
        prediction_service = PredictionService()
        print(f"Model loaded successfully!")
        print(f"Loaded {len(prediction_service.crop_labels)} disease classes")
        
        # Read the image file
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Create a file-like object (mimics Flask's file upload)
        class FakeFile:
            def __init__(self, data):
                self.data = data
                self.position = 0
            
            def read(self):
                return self.data
        
        fake_file = FakeFile(image_data)
        
        # Make prediction
        print("\nMaking prediction...")
        result = prediction_service.predict(fake_file)
        
        # Display results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"\nPredicted Disease: {result['predicted_crop']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        
        print(f"\n Top 3 Predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['crop']}: {pred['confidence']*100:.2f}%")
        
        print("\n" + "="*60)
        print("✓ Test complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nPlant Disease Identifier - Test Script 🌱\n")
    
    # Interactive mode - ask for image path
    while True:
        image_path = input("\nEnter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("\n Goodbye!")
            break
        
        # Remove quotes if user copied path with quotes
        image_path = image_path.strip('"').strip("'")
        
        test_prediction(image_path)
        print("\n" + "-"*60)