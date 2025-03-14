# ใช้ Python 3.9 เป็น base image
FROM python:3.9

# ตั้งค่า Working Directory
WORKDIR /app

# คัดลอกไฟล์ทั้งหมดไปยัง Container
COPY . /app

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

# เปิดพอร์ต 8000
EXPOSE 8000

# รัน Flask API
CMD ["python", "face_api.py"]