import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import PlantInfo from './components/PlantInfo';
import './App.css';

function App() {
  return (
    <div className='App'>
      <Header></Header>

      <div className='mainlayout'>
        <div className='sublayout'>
          <ImageUpload></ImageUpload>
          <PlantInfo></PlantInfo>
        </div>
      </div>
    </div>
  );
}

export default App;
