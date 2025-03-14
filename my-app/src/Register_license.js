import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./Register_license.css";

function Register_license() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [image, setImage] = useState(null);
  const navigate = useNavigate();

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

  // ฟังก์ชันจับภาพจากกล้องแล้วส่งไป API
  const handleRegister = async () => {
    if (!image) {
      alert("กรุณาอัปโหลดรูปภาพก่อน");
      return;
    }

    const formData = new FormData();
    formData.append("image", image);

    try {
      const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";
      const response = await axios.post(`${API_URL}/register_plate`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // The success message from the backend will include the license plate text
      alert(response.data.message);  // This should show "บันทึกป้ายทะเบียน <license_plate> สำเร็จ!"
      navigate("/");
    } catch (error) {
      console.error("❌ เกิดข้อผิดพลาด:", error.response ? error.response.data : error.message);
      alert("เกิดข้อผิดพลาด: " + (error.response ? error.response.data.error : error.message));
    }
  };

  return (
    <div className="background">
      <div className="main-area">
        <h1 className="register-title">ลงทะเบียนด้วยป้ายทะเบียน</h1>
        <p>ระบบจะทำการสแกนป้ายทะเบียนของคุณ...</p>

        {/* Add input for file upload */}
        <div className="input-area">
          <p className="upload-label">Upload License Plate Image</p>
          <input type="file" accept="image/*" onChange={handleImageChange} />
        </div>

        <button className="back-to-main-menu" onClick={handleRegister}>ลงทะเบียน</button>
        <button className="back-to-main-menu" onClick={() => navigate("/")}>
          กลับหน้าหลัก
        </button>
      </div>
    </div>
  );
}

export default Register_license;
