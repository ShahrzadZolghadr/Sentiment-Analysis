FROM python:3.8.10

# install Python modules needed by the Python app
COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

# tell the port number the container should expose
EXPOSE 5000

# run the application
CMD ["python", "app.py"]
