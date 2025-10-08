import './Login.css'
import { useState } from 'react';
import { useNavigate } from 'react-router-dom'

function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const navigate = useNavigate();
    
    const handleUserLogin = () => {
        const username = document.getElementById('name').value;
        const password = document.getElementById('password').value;

        setUsername(username);
        setPassword(password);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        handleUserLogin();

        try {
            const response = await fetch("", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ username, password }),
            });

            const data = await response.json();
            console.log("Login", data);

            navigate('/');
        } catch (error) {
            console.error("Error:", error);
            alert("Error sending data!");
        }
    }
    return (
        <div className="login-container">
            <div className="login-wrapper">
                <h2 className="login-label">Login</h2>
                <form id="loginForm" method="post">
                    <div className="login-layout">
                        <label className="credential-title" ><b>Username</b></label>
                        <input className="credential-value" type="text" placeholder="Enter Username" id="name" name="uname" required></input>
                        
                        <label className="credential-title" ><b>Password</b></label>
                        <input className="credential-value"  type="password" name="psw" id="password" required></input>
                    </div>
                    <button style={{marginTop: '2rem'}} className="submitButton" type="submit" onClick={handleSubmit}>Login</button>
                </form>
            </div>
            
        </div>
    );
}

export default Login