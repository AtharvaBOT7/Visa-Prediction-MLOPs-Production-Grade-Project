FROM python:3.9-slim-buster

WORKDIR /app

# Copy CICD requirements
COPY CICD_requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r CICD_requirements.txt

# Copy application code
COPY . /app

# Expose port
EXPOSE 8080

# Run application
CMD ["python3", "app.py"]