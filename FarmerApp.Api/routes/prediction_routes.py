from flask import Blueprint, request, jsonify
from services.prediction_service import PredictionService
from werkzeug.utils import secure_filename
import os

# Create a Blueprint for prediction routes
prediction_bp = Blueprint('prediction', __name__, url_prefix='/api')

# Initialize the prediction service (singleton pattern ensures model loads once)
prediction_service = PredictionService()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}


def allowed_file(filename):
    """
    Check if uploaded file has an allowed extension
    
    Args:
        filename (str): Name of the uploaded file
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict crop from uploaded image #TODO We can tweak inputs and outputs based on what we need
    
    Expected request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: image file with key 'image'
        
    Returns:
        JSON response with prediction results or error message
        
    Example success response: #TODO: this is only an example of the response, so we can tweak as we need
        {
            "success": true,
            "data": {
                "predicted_crop": "Rice",
                "confidence": 0.95,
                "top_predictions": [...]
            }
        }
        
    Example error response:
        {
            "success": false,
            "error": "No image file provided"
        }
    """
    try:
        # Check if image is in the request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        # Check if file was actually selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Make prediction
        print(f"Processing image: {file.filename}")
        result = prediction_service.predict(file)
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        print(f"Error in prediction endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@prediction_bp.route('/health', methods=['GET'])
def health_check():
    try:
        model_loaded = prediction_service.model_loader.is_model_loaded()
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loaded,
            'message': 'Prediction service is ready' if model_loaded else 'Model not loaded yet'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'message': str(e)
        }), 500


@prediction_bp.route('/supported-crops', methods=['GET'])
def get_supported_crops():
    """
    Get list of crops that the model can identify #TODO: I think this is helpful to let the front end person know what crops are supported. 
    
    Returns:
        JSON response with list of supported crops
        
    Example response:
        {
            "crops": ["Rice", "Wheat", "Maize", "Cotton", ...]
        }
    """
    try:
        return jsonify({
            'crops': prediction_service.crop_labels
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500