/* Added PlantInfo

    component for displaying plant info onto page
*/

import { useState, useEffect } from "react";
import './PlantInfo.css'

function PlantInfo({ plantId }) {
    const [plant, setPlant] = useState({
        'name': "",
        'description': "",
        'confidence': null,
    })

    useEffect(() => {
        if (!plantId) return;
        const getPlantInfo = async () => {
            try {
                const response = await fetch(``);
                if(!response.ok) {throw new Error("Failed to fetch plant data");
                }
                
                const data = await response.json();

                setPlant({
                    'name': data.name,
                    'description': data.description,
                    'confident': data.confidence,
                });

            } catch (err) {
                console.error("Error fetching plant data: ", err);
            }
        }
        getPlantInfo();
    }, [plantId]);

    return(
        <div className="PlantInfo">
            <div className="confident-match">
                <label className="subtitle">Confident Level</label>
                <span style={{marginBottom: '0.5rem'}}>{plant.condident}</span>
           </div>

            <div className="item">
                <label className="subtitle">Name of Plant</label>
                <span>{plant.name}</span>
            </div>

            <div className="item">
                <label className="subtitle">Description of Plant</label>
                <span>{plant.name}</span>
            </div>

        </div>
    );
}

export default PlantInfo