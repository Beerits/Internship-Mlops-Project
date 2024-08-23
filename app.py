

from flask import Flask, render_template, request
import os
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a flask app

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # reading the inputs given by the user
            email = request.form['email']
            obj = PredictionPipeline()
            # obj.transform([email])  # Fit the vectorizer with the provided input
            predict = obj.predict(email)
            return render_template('results.html', prediction=str(predict))
        except Exception as e:
            print(f"Error occurred: {e}")
            return f"An error occurred: {e}"
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.debug = True  # enable debug mode
    app.run(host='0.0.0.0', port=5000)  # run the app
