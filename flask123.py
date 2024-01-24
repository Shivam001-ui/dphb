import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import logging
import requests
from bs4 import BeautifulSoup

# Initialize the flask App
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load the model
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

# Load the model
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    website_url = request.form.get('website_url')
    if not website_url:
        return jsonify({'error': 'No website_url provided.'}), 400

    logger.info(f"Received request for prediction with website URL: {website_url}")

    # Scrape the website
    try:
        response = requests.get(website_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        web_content = soup.prettify()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch the website. Error: {e}")
        return jsonify({'error': 'Failed to scrape the website.'}), 500

    # Use the model to make a prediction
    try:
        prediction = model.predict([web_content]) # Assuming the model expects a list of lists
    except Exception as e:
        logger.error(f"Failed to make a prediction. Error: {e}")
        return jsonify({'error': 'Failed to make a prediction.'}), 500

    return jsonify({'prediction': prediction[0]})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
