import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./Register_face.css";

function Register_face() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [surname, setSurname] = useState("");
  const [image, setImage] = useState(null);

  // ✅ ฟังก์ชันจับไฟล์ภาพ
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
    }
  };

  // ✅ ฟังก์ชันส่งข้อมูลไป API
  const handleRegister = async () => {
    if (!name || !surname || !image) {
      alert("กรุณากรอกชื่อ นามสกุล และอัปโหลดรูปภาพ");
      return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("surname", surname);
    formData.append("image", image);

    try {
      const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";
      const response = await axios.post(`${API_URL}/register_face`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      alert(response.data.message);
      navigate("/");
    } catch (error) {
      console.error("API Error:", error.response ? error.response.data : error.message);
      alert("เกิดข้อผิดพลาด: " + (error.response ? error.response.data.error : error.message));
    }
  };

  return (
    <div className="background">
      <div className="main-area">
        <h1 className="register-title">ยืนยันตัวตนด้วยใบหน้า</h1>
        <div className="input-area">
          <p className="name-register">Upload Image</p>
          <input type="file" accept="image/*" onChange={handleImageChange} />
          <p className="name-register">Name</p>
          <input type="text" placeholder="ชื่อ" value={name} onChange={(e) => setName(e.target.value)} />
          <p className="name-register">Last Name</p>
          <input type="text" placeholder="นามสกุล" value={surname} onChange={(e) => setSurname(e.target.value)} />
        </div>
        <button className="back-to-main-menu" onClick={handleRegister}>ลงทะเบียน</button>
        <button className="back-to-main-menu" onClick={() => navigate("/")}>กลับหน้าหลัก</button>
      </div>
    </div>
  );
}

export default Register_face;
