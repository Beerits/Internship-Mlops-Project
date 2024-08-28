FROM  python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt stopwords
RUN python -m nltk.downloader punkt_tab
CMD ["python", "app.py"]


# # Run app.py when the container launches
# CMD ["python", "app.py"]
