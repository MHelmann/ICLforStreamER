import sys
import logging

sys.path.append('../')
from flask import Flask, request, jsonify
from classifier.model2 import get_model2

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Classifier app2 shutting down...'


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json(force=True)
        entity_pairs_series = data["entity_pairs"]
        model = get_model2()

        predicted_class, confidence = model.predict(entity_pairs_series)
        confidence = confidence.tolist()[0]

        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })


def execute_classifier_app2():
    app.run(host="127.0.0.1", port=5002)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002)
