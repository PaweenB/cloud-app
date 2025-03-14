import React from "react";
import { Link } from "react-router-dom";
import "./Authenpage.css";

function Authenpage() {
  return (
    <div className="background">
      <div className="main-area">
        <h1 className="authen-title">ยืนยันตัวตน</h1>
        <h2 className="authen-title">เลือกวิธีการยืนยัน</h2>
        <div className="flex flex-col items-center gap-4">
          <Link to="/authen/authen_face">
            <button className="authen-face">ใบหน้า</button>
          </Link>
          <Link to="/authen/authen_license">
          <button className="authen-license-plate">ป้ายทะเบียน</button>
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

export default Authenpage;
