import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import PlantInfo from './components/PlantInfo';
import SupportedPlants from './components/SupportedPlants';
import './App.css';
import { useState } from 'react';


function App() {
  const [predictionData, setPredictionData] = useState(null);
  const [showSupportedPlants, setShowSupportedPlants] = useState(false);

  const handlePredictionResult = (data) => {
    setPredictionData(data);
  }

  const handleShowSupportedPlants = () => {
    setShowSupportedPlants(true);
  }

  const handleCloseSupportedPlants = () => {
    setShowSupportedPlants(false);
  }

  return (
    <div className='App'>
      <Header onShowSupportedPlants={handleShowSupportedPlants}></Header>

      <main className='mainlayout'>
        <div className='sublayout'>
          <ImageUpload onPredictionResult={handlePredictionResult}></ImageUpload>
          <PlantInfo predictionData={predictionData}></PlantInfo>
        </div>
      </main>

      <SupportedPlants
        isOpen={showSupportedPlants}
        onClose={handleCloseSupportedPlants}>
      </SupportedPlants>
    </div>
  );
}

export default App;
