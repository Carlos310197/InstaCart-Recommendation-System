from flask import Flask
from routes.health import health_check
from routes.predict import predict

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return health_check()

@app.route('/predict', methods=['POST'])
def prediction():
    return predict()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)