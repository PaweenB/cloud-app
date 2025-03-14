import os
import cv2
import psycopg2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from deepface import DeepFace
from PIL import Image, ImageTk

# 💪 ตั้งค่าการเชื่อมต่อ PostgreSQL
DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "yourpassword",
    "host": "localhost",
    "port": "5432"
}

# 💊 ฟังก์ชันเชื่อมต่อฐานข้อมูล
def connect_db():
    return psycopg2.connect(**DB_PARAMS)

# 🏢 ฟังก์ชันสร้างฐานข้อมูล
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
        )
    ''')
    conn.commit()
    conn.close()

# 🔍 ฟังก์ชันตรวจจับใบหน้าในรูปภาพ
def detect_face(file_path):
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT name, surname, encoding FROM faces")
        data = cursor.fetchall()
        conn.close()

        # ใช้ DeepFace ตรวจจับใบหน้า
        face = DeepFace.represent(file_path, model_name="ArcFace")[0]['embedding']
        target_encoding = np.array(face, dtype=np.float64)

        matched_faces = []
        for name, surname, encoding_bytes in data:
            stored_encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
            distance = np.linalg.norm(target_encoding - stored_encoding)
            if distance < 4.3:
                matched_faces.append(f"{name} {surname}")
        
        if matched_faces:
            messagebox.showinfo("พบข้อมูล", f"พบข้อมูลของ: {', '.join(matched_faces)}")
        else:
            messagebox.showwarning("ไม่พบข้อมูล", "ไม่พบข้อมูลในฐานข้อมูล")
    except Exception as e:
        messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาด: {str(e)}")

# 👤 ฟังก์ชันลงทะเบียนใบหน้า
def register_face(name, surname, file_path):
    if not name or not surname:
        messagebox.showerror("ข้อผิดพลาด", "กรุณากรอกชื่อและนามสกุล")
        return

    img = cv2.imread(file_path)
    img_path = os.path.join("face_database", f"{name}_{surname}.jpg")
    cv2.imwrite(img_path, img)

    try:
        embedding = DeepFace.represent(file_path, model_name="ArcFace")[0]['embedding']
    except:
        messagebox.showerror("ข้อผิดพลาด", "ไม่สามารถตรวจจับใบหน้าได้")
        return

    encoding = np.array(embedding, dtype=np.float64).tobytes()
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO faces (name, surname, image_path, encoding) VALUES (%s, %s, %s, %s)",
                   (name, surname, img_path, encoding))
    conn.commit()
    conn.close()

    messagebox.showinfo("สำเร็จ", f"บันทึก {name} {surname} สำเร็จ!")

# 🎨 สร้าง GUI ด้วย Tkinter
def add_image():
    name = entry_name.get()
    surname = entry_surname.get()

    if not name or not surname:
        messagebox.showerror("ข้อผิดพลาด", "กรุณากรอกชื่อและนามสกุล")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    confirm = messagebox.askyesno("ยืนยัน", f"ต้องการบันทึก {name} {surname} หรือไม่?")
    if confirm:
        register_face(name, surname, file_path)

def search_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    detect_face(file_path)

# 📝 เรียกใช้การสร้างฐานข้อมูล
create_database()

# 🏠 ตั้งค่า GUI
root = tk.Tk()
root.title("Face Recognition App")

tk.Label(root, text="ชื่อ:").grid(row=0, column=0, padx=5, pady=5)
entry_name = tk.Entry(root)
entry_name.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="นามสกุล:").grid(row=1, column=0, padx=5, pady=5)
entry_surname = tk.Entry(root)
entry_surname.grid(row=1, column=1, padx=5, pady=5)

tk.Button(root, text="เพิ่มรูปภาพ", command=add_image).grid(row=2, column=0, columnspan=2, pady=5)
tk.Button(root, text="ตรวจสอบจากไฟล์", command=search_image).grid(row=3, column=0, columnspan=2, pady=5)

root.mainloop()
