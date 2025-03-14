import React from "react";
import { Link } from "react-router-dom";
import "./Registerpage.css";

function Registerpage() {
  return (
    <div className="background">
      <div className="main-area">
        <h1 className="registerpage-title">ลงทะเบียน</h1>
        <h2 className="registerpage-title">เลือกวิธีการยืนยัน</h2>
        <div className="flex flex-col items-center gap-4">
          <Link to="/register/register_face">
          <button className="registerpage-face">ใบหน้า</button>
          </Link>
          <Link to="/register/register_license">
          <button className="registerpage-license-plate">ป้ายทะเบียน</button>
          </Link>
          {/* Link to navigate back to the main page */}
          <Link to="/">
            <button className="back-to-main-menu">กลับหน้าหลัก</button>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Registerpage;
