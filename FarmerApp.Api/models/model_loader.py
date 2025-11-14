import tensorflow as tf
import os
from typing import Optional

class ModelLoader:   
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str = "models/crop_model.keras"):
        if self._model is not None:
            print("Model already loaded, returning cached version")
            return self._model
    
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
    
        try:
            print(f"Loading model from {model_path}...")
            self._model = tf.keras.models.load_model(model_path, compile=False)
            self._model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("Model loaded successfully!")
            return self._model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def get_model(self):
        if self._model is None:
            self.load_model()
        return self._model
    
    def is_model_loaded(self) -> bool:
        return self._model is not None