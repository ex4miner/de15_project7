FROM python:3.10-slim

COPY requirements.txt /requirements.txt
    
COPY . /app
    
WORKDIR /

ENV FLASK_APP app/main.py
    
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
    
ENTRYPOINT ["python", "-m", "flask", "run", "--host=0.0.0.0", "--reload"]