# Sử dụng Python slim image
FROM python:3.9-slim

# Cài đặt thư mục làm việc
WORKDIR /app

# Cài đặt các dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code frontend vào container
COPY . .

# Mở cổng mặc định của Streamlit
EXPOSE 8501

# Chạy ứng dụng Streamlit
CMD ["streamlit", "run", "Hello.py", "--server.port=8501", "--server.address=0.0.0.0"]
