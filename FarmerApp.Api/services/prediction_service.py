import numpy as np
from PIL import Image
import io
import json
import os
from models.model_loader import ModelLoader
from typing import Dict, Any


class PredictionService:   
    def __init__(self):
        # Load the trained model
        self.model_loader = ModelLoader()
        self.model = self.model_loader.get_model()
        
        self.img_height = 224 
        self.img_width = 224
        
        # Load class labels from JSON file
        self.crop_labels = self._load_class_labels()
    
    def _load_class_labels(self):
        labels_path = "models/class_labels.json"
        
        try:
            if not os.path.exists(labels_path):
                print(f"Warning: {labels_path} not found, using default labels")
                return [
                    'Rice', 'Wheat', 'Maize', 'Cotton',
                    'Tomato', 'Potato', 'Sugarcane', 'Soybean'
                ]
            
            with open(labels_path, 'r') as f:
                labels_dict = json.load(f)
            
            num_classes = len(labels_dict)
            labels_list = [None] * num_classes
            
            for idx_str, label_name in labels_dict.items():
                idx = int(idx_str)
                labels_list[idx] = label_name
            
            print(f"Loaded {len(labels_list)} class labels from {labels_path}")
            return labels_list
            
        except Exception as e:
            print(f"Error loading class labels: {str(e)}")
            print("Using default labels as fallback")
            return [
                'Rice', 'Wheat', 'Maize', 'Cotton',
                'Tomato', 'Potato', 'Sugarcane', 'Soybean'
            ]

    def preprocess_image(self, image_file) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_file.read()))
            image = image.convert('RGB')
            image = image.resize((self.img_width, self.img_height))
        
            img_array = np.array(image)
        
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