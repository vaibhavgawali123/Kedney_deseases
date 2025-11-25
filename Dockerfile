# Use Python base image
FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update -y && apt-get install -y libgl1-mesa-glx

# Copy requirement file
COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy backend files
COPY app.py .
COPY model_converted.h5 ./model_converted.h5

# Copy any other model files
# COPY model.h5 ./model.h5
# COPY model.pkl ./model.pkl

# Expose port
EXPOSE 10000

# Run the app
CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
