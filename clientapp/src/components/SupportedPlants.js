import './SupportedPlants.css'
import { useState, useEffect } from "react";

function SupportedPlants({ isOpen, onClose }) {
    const [crops, setCrops] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (isOpen) {
            fetchSupportedCrops();
        }
    }, [ isOpen ]);

    const fetchSupportedCrops = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/supported-crops');

            if (!response.ok) {
                throw new Error('Failed to fetch supported crops');
            }

            const data = await response.json();
            setCrops(data.crops || []);
        } catch (err) {
            console.error("Error fetching supported crops:", err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className='modal-overlay' onClick={onClose}>
            <div className='modal-content' onClick={(e) => e.stopPropagation()}>
                <div className='modal-header'>
                    <h2>Supported Plants</h2>
                    <button className='close-btn' onClick={onClose}>X</button>
                </div>

                <div className='modal-body'>
                    {loading && <p>Loading...</p>}
                    {error && <p className='error-message'>Error: {error}</p>}
                    {!loading && !error && (
                        <ul className='crops-list'>
                            {crops.map((crop, index) => (
                                <li key={index} className='crop-item'>
                                    {crop}
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            </div>
        </div>
    );
}

export default SupportedPlants