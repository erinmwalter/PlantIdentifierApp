import { useState } from "react";
import './SignUp.css'

function SignUp() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');

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
        } catch (error) {
            console.error("Error:", error);
            alert("Error sending data!");
        }
    }

    return (
        <div className="signup-container">
            <div className="signup-wrapper">
                <h2 className="signup-label">SignUp</h2>
                <form id="signupForm" method="post">
                    <div className="signup-layout">
                        <label className="credential-title" ><b>Username</b></label>
                        <input className="credential-value" type="text" placeholder="Enter Username" id="name" name="uname" required></input>
                        
                        <label className="credential-title" ><b>Password</b></label>
                        <input className="credential-value" type="password" name="psw" id="password" required></input>
                        
                        <label className="credential-title" ><b>Confirm Password</b></label>
                        <input className="credential-value"  type="password" name="psw" id="password" required></input>
                    </div>
                    <button style={{marginTop: '2rem'}}className="submitButton" type="submit" onClick={handleSubmit}>SignUp</button>
                </form>
                <p>
                    Already have an account? <a href="/login">Login</a>
                </p>
            </div>
            
        </div>
    );
}

export default SignUp