from flask import Flask, request, render_template, jsonify
from predict import *
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict_api', methods=['GET', 'POST'])
def predict_classes():
    if request.method == 'GET':
        return render_template('home.html', value = "Image")
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Image not uploaded"
        file = request.files['file'].read()

        try:
            img = Image.open(io.BytesIO(file))
        except IOError:
            return jsonify(predictions = "Not an Image, Upload a proper image file", preds_prob = "")
        
        img = img.convert("RGB")
        p = predict(img)

        return jsonify(predicitions = p)

if __name__ == '__main__':
    app.run(debug = True)
