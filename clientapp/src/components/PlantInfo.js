/* Added PlantInfo

    component for displaying plant info onto page
*/

//import { useState } from "react";
import './PlantInfo.css'

function PlantInfo() {
    fetch('', {
        method: 'GET',
    })
        .then(response => response.json())
        .then()

    return(
        <div className="PlantInfo">
            <div className="confident-match">
                <label className="subtitle">Confident Level</label>
                <span style={{marginBottom: '0.5rem'}}>85%</span>
           </div>

            <div className="item">
                <label className="subtitle">Name of Plant</label>
                <span>Placeholder</span>
            </div>

            <div className="item">
                <label className="subtitle">Description of Plant</label>
                <span>Placeholder</span>
            </div>

        </div>
    );
}

export default PlantInfo