# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies required for Python packages


# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --upgrade openai
pip install langchain
# Copy the entire project to the working directory
COPY . .

# Ensure the ChromaDB persistent storage directory exists


# Expose the port the Flask app runs on
EXPOSE 8000

# Set environment variables from .env file (optional, can be passed via docker run)
# Alternatively, you can use a .env file with docker-compose or pass them at runtime
ENV MONGODB_URI=$MONGODB_URI
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Command to run the Flask application
CMD ["python", "serve.py"]