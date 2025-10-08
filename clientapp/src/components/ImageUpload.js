/* Added ImageUpload function
    
    component for uploading images to the website
*/

import { useState } from "react";
import './ImageUpload.css'

function ImageUpload() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);

    const handleSubmit = (event) => {
        event.preventDefault();
        try {
        
        // const response = await fetch("", {
        //     method: "POST",
        //     headers: {
        //         "Content-Type": "application/json",
        //     },
        //     body: JSON.stringify(file),
        // });

        // const data = await response.json();
        // console.log("Success:", data);
        // alert("Data submitted successfully!");
        
        // if (response.ok) {
        //     alert("Post created successfully!");
        //     setFile("");
        // } else {
        //     alert(data.message || "Something went wrong!");
        // } 
        } catch(error) {
            // console.error("Error:", error);
            // alert("Error sending data!");
        }
    };
    
    const handleUpload = (event) => {
        const image = event.target.files[0];

        if(image) {
            setFile(image);
            const previewUrl = URL.createObjectURL(image);
            setPreview(previewUrl);
            console.log('File Name:', image.name);
        }
    }

    const openFile = () => {
        document.getElementById("plant-upload").click();
    }

    const clearUpload = () => {
        setFile(null);
        setPreview(null);
        document.getElementById('plant-upload').value = "";
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
                    <button onClick={openFile} className="UploadButton">
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