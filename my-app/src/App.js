import React from "react";
import { Link } from "react-router-dom"; // Import Link from React Router
import "./App.css";

function App() {
  return (
    <div className="background">
      <div className="main-area">
        <h1 className="my-app-title">my app</h1>
        
        <div className="flex flex-col items-center gap-4">
          {/* Use Link to navigate to Authenpage */}
          <Link to="/authen">
            <button className="authen-button">ยืนยันตัวตน</button>
          </Link>
          <Link to="/register">
          <button className="register-button">ลงทะเบียน</button>
            </Link>
        </div>
      </div>
    </div>
  );
}

export default App;
