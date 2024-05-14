FROM python:3.8-slim

WORKDIR /app

COPY . /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN pip install -r /app/requirements.txt

CMD ["streamlit", "run", "/app/main.py"]