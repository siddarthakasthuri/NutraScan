# app.py
from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from food_detection_model import detect_food, get_calorie_count, get_ingredients
import os
import tempfile
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# app.py
import os
import tempfile

@app.route('/upload', methods=['POST'])
def upload():
    # Get base64 encoded image data from the request
    image_data = request.form['image']

    # Decode the image
    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(cv2.imencode('.jpg', img)[1])
        image_path = temp_file.name

    # Process the image
    food_item = detect_food(image_path)
    calorie_count = get_calorie_count(image_path)
    food_ingredients = get_ingredients(image_path)

    # Remove the temporary file
    os.unlink(image_path)

    # Return the detected food item and its calorie count
    return jsonify(food_item=food_item, calorie_count=calorie_count, food_ingredients=food_ingredients)

if __name__ == '__main__':
    app.run(debug=True)
