/* Added ImageUpload function
    
    component for uploading images to the website
*/

import { useState } from "react";
import './ImageUpload.css'

function ImageUpload() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);

    // CLEARING UPLOAD
    const clearUpload = () => {
        setFile(null);
        setPreview(null);
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
            console.log('File Name: ', image.name);
        }
    }

    // HANDLES SUBMIT AND SENDS TO BACKEND
    const handleSubmit = async (event) => {
        event.preventDefault();
        try {
            const formData = new FormData();
            formData.append('image', file);
            const response = await fetch(``, {      // TODO: ENTER BACKEND API
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error("failed to upload image");
            }

            if (file) {

            }
        } catch (error) {
            console.error("Error upload image: ", error);
        }
    }



    return(
        <div className="ImageUpload">
            {preview && <span className="clearButton" onClick={clearUpload}>Clear</span>}
            {preview && 
            <img src={preview} alt="preview" width="200" height="200">
            </img>
            }
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
                    <button type="submit" className="SubmitButton">
                        SUBMIT
                    </button>
                </div>
            </form>
            
        </div>
    );
}

export default ImageUpload