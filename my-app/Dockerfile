# ใช้ Node.js เป็น base image
FROM node:18

# ตั้งค่า Working Directory
WORKDIR /app

# คัดลอกไฟล์ไปยัง Container
Copy . /app

# ติดตั้ง dependencies
RUN npm install

# คัดลอกโค้ดทั้งหมดไปยัง Container
COPY . .

# เปิดพอร์ต 3000
EXPOSE 3000

# รันแอป (React: npm start, Vue: npm run serve, Next.js: npm run dev)
CMD ["npm", "start"]