import './Header.css'
function Header({ onShowSupportedPlants}) {
    return (
        <header className="Header">
            <span className='Title'>PLANT IDENTIFICATION</span>

            <button
                className='supported-plants-btn'
                onClick={onShowSupportedPlants}>
                SUPPORTED PLANTS
            </button>
        </header>
    );

}

export default Header