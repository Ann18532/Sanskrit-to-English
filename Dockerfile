# Use the official TensorFlow GPU image as a base
FROM tensorflow/tensorflow:2.13.0-gpu

# Set the working directory in the container
WORKDIR /app

# Add the working directory to PYTHONPATH
ENV PYTHONPATH=/app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the default command to run the script
# Ensure the script path is correct relative to the WORKDIR
CMD ["python", "utils/sandhi/main.py"] 