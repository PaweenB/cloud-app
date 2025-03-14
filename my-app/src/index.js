import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import App from "./App";
import Authenpage from "./Authenpage";
import Registerpage from "./Registerpage";
import Authen_face from "./Authen_face";
import Authen_license from "./Authen_license";
import Register_face from "./Register_face";
import Register_license from "./Register_license";
import "./index.css";

const root = ReactDOM.createRoot(document.getElementById("root"));

root.render(
  <React.StrictMode>
    <Router>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/authen" element={<Authenpage />} />
        <Route path="/register" element={<Registerpage />} />
        <Route path="/authen/authen_face" element={<Authen_face />} />
        <Route path="/authen/Authen_license" element={<Authen_license />} />
        <Route path="/register/register_face" element={<Register_face />} />
        <Route path="/register/register_license" element={<Register_license />} />
      </Routes>
    </Router>
  </React.StrictMode>
);
