FROM python:3.10-slim-buster

WORKDIR /app

COPY CICD_requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r CICD_requirements.txt

COPY . /app

EXPOSE 8080

CMD ["python3", "app.py"]