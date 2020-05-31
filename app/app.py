import argparse

import flask
from flask import Flask, jsonify, request
import requests

from utils import load_model, load_tokenizer, encode_text, predict
from transformers import AutoTokenizer

app = Flask(__name__)

Model, config_file = load_model("./src/Baseline_model/state_dict.py", "./src/Baseline_model/config_model.json") # argparse
tokenizer = load_tokenizer()

@app.route('/predict', methods = ['POST'])
def api_sentiment():
    data = request.get_json(force = True)

    encoded = encode_text(data['text'], tokenizer, config_file)

    prediction = predict(Model, encoded)

    request_responses = {
        "Negative Score" : prediction[0][0],
        "Positive Score" : prediction[0][1]
    }

    return jsonify(request_responses)

if __name__ == "__main__":
    
 """    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_model", help="Path to the pre trained model", typr=str)
    parser.add_argument("--path_to_config", help="Path to the config file", type=str)

    args = parser.parse_args()
 """
    app.run(debug=True)
    





