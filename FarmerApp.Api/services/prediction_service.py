import numpy as np
from PIL import Image
import io
from models.model_loader import ModelLoader
from typing import Dict, Any


class PredictionService:   
    def __init__(self):
        # Load the trained model
        self.model_loader = ModelLoader()
        self.model = self.model_loader.get_model()
        
        # Image input size TODO: Fix this based on how model is trained?
        self.img_height = 224 
        self.img_width = 224
        
        # TODO: Update this list with the actual crops from training
        # I put some placeholder crops here just for the sake of getting code flow
        # written but this can be changed as needed.
        self.crop_labels = [
            'Rice', 
            'Wheat', 
            'Maize',
            'Cotton',
            'Tomato',
            'Potato',
            'Sugarcane',
            'Soybean'
            # Add more crops as needed
        ]
    
    def preprocess_image(self, image_file) -> np.ndarray:
        try:
            # Read the image file
            image = Image.open(io.BytesIO(image_file.read()))
            
            # Convert to RGB (handles RGBA, grayscale, etc.)
            image = image.convert('RGB')
            
            # Resize to model's expected input size
            image = image.resize((self.img_width, self.img_height))
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Normalize pixel values to [0, 1] range
            img_array = img_array / 255.0
            
            # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def get_crop_name(self, prediction_index: int) -> str:
        if 0 <= prediction_index < len(self.crop_labels):
            return self.crop_labels[prediction_index]
        else:
            return "Unknown"
    
    def predict(self, image_file) -> Dict[str, Any]:
        try:
            # Step 1: Preprocess the image
            print("Preprocessing image...")
            processed_image = self.preprocess_image(image_file)
            
            # Step 2: Make prediction using the model
            print("Making prediction...")
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Step 3: Get the predicted class index (highest probability)
            predicted_index = np.argmax(predictions[0])
            
            # Step 4: Get confidence score for the predicted class
            confidence = float(predictions[0][predicted_index])
            
            # Step 5: Get the crop name
            predicted_crop = self.get_crop_name(predicted_index)
            
            # Step 6: Get top 3 predictions for additional context
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]  # Get top 3 in descending order
            top_predictions = [
                {
                    'crop': self.get_crop_name(idx),
                    'confidence': float(predictions[0][idx])
                }
                for idx in top_3_indices
            ]
            
            print(f"Prediction complete: {predicted_crop} ({confidence:.2%})")
            
            return {
                'predicted_crop': predicted_crop,
                'confidence': confidence,
                'top_predictions': top_predictions
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")