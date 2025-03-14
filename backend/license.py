import os
import psycopg2
from tkinter import Tk, Button, Label, filedialog, messagebox
import cv2
import easyocr
from shutil import copyfile
from tkinter.simpledialog import askstring
import difflib

# 🔹 ตั้งค่าการเชื่อมต่อ PostgreSQL
DB_PARAMS = {
    "dbname": "face_db",
    "user": "postgres",
    "password": "yourpassword",
    "host": "10.153.37.203",
    "port": "5432"
}

# 🔹 สร้างโฟลเดอร์สำหรับเก็บภาพป้ายทะเบียน
PLATE_DIR = "plate_database"
os.makedirs(PLATE_DIR, exist_ok=True)

# 🔹 ฟังก์ชันเชื่อมต่อฐานข้อมูล
def connect_db():
    try:
        return psycopg2.connect(**DB_PARAMS)
    except psycopg2.Error as e:
        messagebox.showerror("Database Error", f"ไม่สามารถเชื่อมต่อฐานข้อมูล: {e}")
        return None

# 🔹 สร้างตารางในฐานข้อมูล (ถ้ายังไม่มี)
def create_database():
    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS plates (
                    id SERIAL PRIMARY KEY,
                    license_plate TEXT NOT NULL,
                    image_path TEXT NOT NULL
                )
            ''')
            conn.commit()
        except psycopg2.Error as e:
            messagebox.showerror("Database Error", f"ไม่สามารถสร้างตาราง: {e}")
        finally:
            conn.close()

# 🔹 ฟังก์ชันสำหรับอ่านข้อความจากป้ายทะเบียน
def read_license_plate(image_path):
    try:
        # สร้าง Reader object สำหรับ OCR (ภาษาไทย + อังกฤษ)
        reader = easyocr.Reader(['th', 'en'])

        # OCR อ่านตัวอักษรจากภาพ (ตั้งค่า detail=0 ให้คืนค่าเป็นข้อความล้วน)
        results = reader.readtext(image_path, detail=0)

        # รวมข้อความที่อ่านได้
        license_plate_text = " ".join(results)

        return license_plate_text.strip()
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการอ่านป้ายทะเบียน: {e}")
        return None

# 🔹 ฟังก์ชันตรวจสอบว่ามีป้ายทะเบียนซ้ำในฐานข้อมูลหรือไม่
def check_duplicate_plate(license_plate_text):
    try:
        conn = connect_db()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM plates WHERE license_plate = %s", (license_plate_text,))
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0  # ถ้า count > 0 แปลว่ามีอยู่แล้ว
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการตรวจสอบข้อมูลซ้ำ: {e}")
        return False

# 🔹 ฟังก์ชันสำหรับบันทึกข้อมูลลงในฐานข้อมูล PostgreSQL
def save_to_database(license_plate_text, image_path):
    if check_duplicate_plate(license_plate_text):
        messagebox.showwarning("แจ้งเตือน", f"ป้ายทะเบียน '{license_plate_text}' มีอยู่ในระบบแล้ว!")
        return

    try:
        conn = connect_db()
        if conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO plates (license_plate, image_path)
                VALUES (%s, %s)
            ''', (license_plate_text, image_path))
            conn.commit()
            conn.close()
            messagebox.showinfo("บันทึกสำเร็จ", f"บันทึกป้ายทะเบียน '{license_plate_text}' เรียบร้อยแล้ว")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการบันทึกข้อมูล: {e}")
        return f"เกิดข้อผิดพลาดในการบันทึกข้อมูล: {e}"

# 🔹 ฟังก์ชันสำหรับเลือกรูปภาพ
def select_image():
    file_path = filedialog.askopenfilename(
        title="เลือกรูปภาพ",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )

    if file_path:
        license_plate = read_license_plate(file_path)
        if license_plate:
            # บันทึกข้อมูลในฐานข้อมูล
            message = save_to_database(license_plate, file_path)
            messagebox.showinfo("ผลลัพธ์", f"ข้อความที่ได้จากป้ายทะเบียน:\n{license_plate}\n{message}")
        else:
            messagebox.showerror("ข้อผิดพลาด", "ไม่สามารถอ่านข้อความจากป้ายทะเบียนได้")
    else:
        messagebox.showwarning("แจ้งเตือน", "ไม่มีการเลือกรูปภาพ")

# 🔹 ฟังก์ชันสำหรับตรวจสอบข้อมูลป้ายทะเบียนจากภาพ
def check_license_plate():
    # ให้ผู้ใช้เลือกไฟล์ภาพ
    file_path = filedialog.askopenfilename(
        title="เลือกรูปภาพ",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )

    if file_path:
        # อ่านข้อความจากภาพ
        license_plate_text = read_license_plate(file_path)
        if license_plate_text:
            conn = connect_db()
            if conn:
                try:
                    cursor = conn.cursor()
                    # ค้นหาข้อมูลทั้งหมดในฐานข้อมูล
                    cursor.execute("SELECT * FROM plates")
                    result = cursor.fetchall()

                    # เปรียบเทียบข้อความจากภาพกับป้ายทะเบียนในฐานข้อมูล
                    found = False
                    for row in result:
                        db_license_plate = row[1]
                        # คำนวณความคล้ายกันของข้อความด้วย difflib
                        similarity = difflib.SequenceMatcher(None, license_plate_text, db_license_plate).ratio()

                        if similarity > 0.6:
                            messagebox.showinfo("ผลลัพธ์การตรวจสอบ", f"พบข้อมูลป้ายทะเบียน: {db_license_plate}\nข้อมูลที่บันทึกในระบบ\nความคล้าย: {round(similarity * 100, 2)}%")
                            found = True
                            break

                    if not found:
                        messagebox.showinfo("ผลลัพธ์การตรวจสอบ", "ไม่พบข้อมูลป้ายทะเบียนที่คล้ายกันในระบบ กรุณาลงทะเบียน")

                except psycopg2.Error as e:
                    messagebox.showerror("Database Error", f"เกิดข้อผิดพลาดในการตรวจสอบข้อมูล: {e}")
                finally:
                    conn.close()
        else:
            messagebox.showerror("ข้อผิดพลาด", "ไม่สามารถอ่านข้อความจากป้ายทะเบียนได้")
    else:
        messagebox.showwarning("แจ้งเตือน", "ไม่มีการเลือกรูปภาพ")

# 🔹 สร้างหน้าต่าง GUI
def create_gui():
    root = Tk()
    root.title("OCR อ่านป้ายทะเบียนรถ")
    root.geometry("400x200")

    # ปุ่มเลือกภาพ
    btn_select = Button(root, text="เลือกรูปภาพ", command=select_image)
    btn_select.pack(pady=10)

    # ปุ่มตรวจสอบป้ายทะเบียน
    btn_check = Button(root, text="ตรวจสอบป้ายทะเบียน", command=check_license_plate)
    btn_check.pack(pady=10)

    # ป้ายแสดงสถานะ
    label_status = Label(root, text="กรุณาเลือกรูปภาพหรือกรอกป้ายทะเบียนเพื่อตรวจสอบ")
    label_status.pack(pady=10)
    # รัน GUI
    root.mainloop()

# เรียกใช้งานฟังก์ชันต่างๆ
create_database()  # สร้างตารางถ้ายังไม่มี
create_gui()    # สร้าง GUI และรันโปรแกรม