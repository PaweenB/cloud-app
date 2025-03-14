import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./Authen_license.css";

function Authen_license() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const navigate = useNavigate();
  const [image, setImage] = useState(null);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((error) => {
        console.error("ไม่สามารถเข้าถึงกล้องได้:", error);
      });

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // ฟังก์ชันจับไฟล์ภาพ
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
    }
  };

  // ฟังก์ชันจับภาพแล้วส่งไป API
  const handleAuthen = async () => {
    if (!image) {
      alert("กรุณาอัปโหลดรูปภาพก่อน");
      return;
    }

    const formData = new FormData();
    formData.append("image", image);

    try {
      const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";
      const response = await axios.post(`${API_URL}/check_plate`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      alert(response.data.message);
    } catch (error) {
      console.error("❌ เกิดข้อผิดพลาด:", error.response ? error.response.data : error.message);
      alert("เกิดข้อผิดพลาด: " + (error.response ? error.response.data.error : error.message));
    }
  };

  return (
    <div className="background">
      <div className="main-area">
        <h1 className="authen-title">ยืนยันตัวตนด้วยป้ายทะเบียน</h1>
        <p>ระบบจะทำการสแกนป้ายทะเบียนของคุณ...</p>
        {/* Add input for file upload */}
        <div className="input-area">
          <p className="upload-label">Upload License Plate Image</p>
          <input type="file" accept="image/*" onChange={handleImageChange} />
        </div>
        <button className="back-to-main-menu" onClick={handleAuthen}>ยืนยันตัวตน</button>
        <button className="back-to-main-menu" onClick={() => navigate("/")}>
          กลับหน้าหลัก
        </button>
      </div>
    </div>
  );
}

export default Authen_license;
