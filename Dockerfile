# # Use an official Python runtime as a parent image
# FROM python:3.8-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app
# # COPY ./app.py /app/app.py

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# RUN python -m nltk.downloader punkt

# EXPOSE 5000

# # Define environment variable
# # ENV NAME World



FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]


# # Run app.py when the container launches
# CMD ["python", "app.py"]
