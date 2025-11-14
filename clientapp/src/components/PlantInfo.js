/* Added PlantInfo

    component for displaying plant info onto page
*/

import './PlantInfo.css'

function PlantInfo({ predictionData }) {
    if (!predictionData) {
        return (
            <div className="PlantInfo">
                <p>Upload an image to see plant identification results</p>
            </div>
        )
    }

    return(
        <div className="PlantInfo">
            <div className="confident-match">
                <label className="subtitle">Confidence Level</label>
                <span style={{ marginBottom: '0.5rem' }}>
                    {(predictionData.confidence * 100).toFixed(2)}%
                </span>
           </div>

            <div className="item">
                <label className="subtitle">Predicted Crop</label>
                <span>{predictionData.predicted_crop}</span>
            </div>

            {predictionData.top_predictions && (
                <div className="item">
                    <label className="subtitle">Top Predictions</label>
                    <div className="top-predictions">
                        {predictionData.top_predictions.map((pred, index) => (
                            <div key={index} className="prediction-item">
                                <span className="crop-name">{pred.crop}</span>
                                <span className="crop-confidence">
                                    {(pred.confidence * 100).toFixed(2)}%
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

        </div>
    );
}

export default PlantInfo