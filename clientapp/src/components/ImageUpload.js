/* Added ImageUpload function
    
    component for uploading images to the website
*/
import { useState } from "react";
import './ImageUpload.css'

function ImageUpload({ onPredictionResult }) {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // CLEARING UPLOAD
    const clearUpload = () => {
        setFile(null);
        setPreview(null);
        setError(null);
        if (onPredictionResult) {
            onPredictionResult(null);
        }
        document.getElementById('plant-upload').value = "";
    };

    // OPENING UPLOAD
    const openFile = () => {
        document.getElementById('plant-upload').click();
    }

    // HANDLES SETTING IMAGE AND PREVIEW
    const handleUpload = (event) => {
        const image = event.target.files[0];

        if (image) {
            setFile(image);
            const previewURL = URL.createObjectURL(image);
            setPreview(previewURL);
            setError(null);
            console.log('File Name: ', image.name);
        }
    }

    // HANDLES SUBMIT AND SENDS TO BACKEND
    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!file) {
            setError("Please select an image first");
            return;
        }
        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('image', file);
            const response = await fetch(`/api/predict`, {      // TODO: ENTER BACKEND API
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error("failed to upload image");
            }

            if (result.success && onPredictionResult) {
                onPredictionResult(result.data);
            }
        } catch (error) {
            console.error("Error upload image: ", error);
            setError(error.message || "Failed to get prediction");
        } finally {
            setLoading(false);
        }
    }



    return(
        <div className="ImageUpload">
            {preview && <span className="clearButton" onClick={clearUpload}>Clear</span>}
            {preview && 
            <img src={preview} alt="preview" width="200" height="200">
            </img>
            }
            {error && <div className="error-message">{error}</div>}
            <form onSubmit={handleSubmit}>
                <input 
                    style={{display: 'none'}} 
                    type="file" 
                    id="plant-upload" 
                    accept="image/jpeg, image/png, image/jpg" 
                    onChange={handleUpload}>
                </input>
                <div className="buttonlayout">
                    <button type='button' onClick={openFile} className="UploadButton">
                        UPLOAD
                    </button>
                    <button type="submit" className="SubmitButton" disabled={!file || loading}>
                        {loading ? 'PROCESSING...' : 'SUBMIT'}
                    </button>
                </div>
            </form>
            
        </div>
    );
}

export default ImageUpload