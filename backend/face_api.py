from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import psycopg2
from deepface import DeepFace
import easyocr
from tkinter import Tk, Button, Label, filedialog, messagebox
import requests
import cv2
import difflib

# ตั้งค่าการเชื่อมต่อ PostgreSQL
DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "yourpassword",
    "host": "localhost",
    "port": "5432"
}

def connect_db():
    return psycopg2.connect(**DB_PARAMS)

# โฟลเดอร์เก็บภาพ
FACE_DIR = "face_database"
PLATE_DIR = "plate_database"
os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(PLATE_DIR, exist_ok=True)

# สร้างตารางในฐานข้อมูล
def create_database():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            surname TEXT NOT NULL,
            image_path TEXT NOT NULL,
            encoding BYTEA NOT NULL
        )''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id SERIAL PRIMARY KEY,
            license_plate TEXT NOT NULL,
            image_path TEXT NOT NULL
        )''')
    conn.commit()
    conn.close()

app = Flask(__name__)
CORS(app)

@app.route("/register_face", methods=["POST"])
def register_face():
    try:
        name, surname, image = request.form.get("name"), request.form.get("surname"), request.files.get("image")
        if not name or not surname or not image:
            return jsonify({"error": "กรุณากรอกข้อมูลให้ครบ"}), 400
        image_path = os.path.join(FACE_DIR, f"{name}_{surname}.jpg")
        image.save(image_path)
        embedding = DeepFace.represent(image_path, model_name="ArcFace", enforce_detection=False)[0]['embedding']
        encoding = np.array(embedding, dtype=np.float64).tobytes()
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO faces (name, surname, image_path, encoding) VALUES (%s, %s, %s, %s)",
                       (name, surname, image_path, encoding))
        conn.commit()
        conn.close()
        return jsonify({"message": f"ลงทะเบียน {name} {surname} สำเร็จ!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/authen_face", methods=["POST"])
def authen_face():
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "กรุณาอัปโหลดภาพ"}), 400
        image_path = "temp_face.jpg"
        image.save(image_path)
        face = DeepFace.represent(image_path, model_name="ArcFace")[0]['embedding']
        target_encoding = np.array(face, dtype=np.float64)
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT name, surname, encoding FROM faces")
        data = cursor.fetchall()
        conn.close()
        matched_faces = [f"{name} {surname}" for name, surname, enc in data
                         if np.linalg.norm(target_encoding - np.frombuffer(enc, dtype=np.float64)) < 4.3]
        return jsonify({"message": f"พบข้อมูล: {', '.join(matched_faces)}" if matched_faces else "ไม่พบข้อมูล"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/register_plate", methods=["POST"])
def register_plate():
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "กรุณาอัปโหลดภาพ"}), 400
        image_path = os.path.join(PLATE_DIR, image.filename)
        image.save(image_path)
        
        # Use EasyOCR to read the text on the license plate
        reader = easyocr.Reader(['th', 'en'])
        results = reader.readtext(image_path, detail=0)
        license_plate_text = " ".join(results).strip()

        # Store the license plate in the database
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO plates (license_plate, image_path) VALUES (%s, %s)",
                       (license_plate_text, image_path))
        conn.commit()
        conn.close()

        # Send back the success message with the license plate text
        return jsonify({"message": f"บันทึกป้ายทะเบียน {license_plate_text} สำเร็จ!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/check_plate", methods=["POST"])
def check_plate():
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "กรุณาอัปโหลดภาพ"}), 400
        image_path = "temp_plate.jpg"
        image.save(image_path)

        reader = easyocr.Reader(['th', 'en'])
        results = reader.readtext(image_path, detail=0)
        license_plate_text = " ".join(results).strip().upper()  # แปลงเป็นตัวพิมพ์ใหญ่ทั้งหมด

        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT license_plate FROM plates")
        plates = [p[0].upper() for p in cursor.fetchall()]  # แปลงข้อมูลจากฐานข้อมูลให้เป็นตัวพิมพ์ใหญ่
        conn.close()

        # หาค่าความคล้ายสูงสุด
        best_match = None
        best_ratio = 0

        for plate in plates:
            similarity = difflib.SequenceMatcher(None, license_plate_text, plate).ratio()
            if similarity > best_ratio:
                best_match = plate
                best_ratio = similarity

        # กำหนด threshold ที่ 0.8 เพื่อกรองข้อมูลให้แม่นยำขึ้น
        if best_ratio >= 0.8:
            return jsonify({"message": f"พบป้ายทะเบียน: {best_match}"})
        else:
            return jsonify({"message": "ไม่พบข้อมูล"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    create_database()
    app.run(host="0.0.0.0", port=8000, debug=True)

# GUI เชื่อมต่อ API Flask
def select_image(api_endpoint):
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
    if file_path:
        with open(file_path, "rb") as image_file:
            response = requests.post(f"http://localhost:8000/{api_endpoint}", files={"image": image_file})
        messagebox.showinfo("ผลลัพธ์", response.json().get("message", "เกิดข้อผิดพลาด"))

def create_gui():
    root = Tk()
    root.title("ระบบ OCR และจดจำใบหน้า")
    Button(root, text="ลงทะเบียนใบหน้า", command=lambda: select_image("register_face")).pack(pady=5)
    Button(root, text="ตรวจสอบใบหน้า", command=lambda: select_image("authen_face")).pack(pady=5)
    Button(root, text="ลงทะเบียนป้ายทะเบียน", command=lambda: select_image("register_plate")).pack(pady=5)
    Button(root, text="ตรวจสอบป้ายทะเบียน", command=lambda: select_image("check_plate")).pack(pady=5)
    root.mainloop()

create_gui()
