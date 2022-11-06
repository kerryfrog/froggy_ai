from flask import Flask, jsonify
from flask_restful import reqparse
import joblib

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/recommend/pattern')
def pattern():
    return "ppatternRouter"

@app.route('/recommend', methods=['POST'])
def predict():
    parser = reqparse.RequestParser()
    data = {
        'test' : 'get data from flask server!',
        'som' : 'dasom'
    }
    return jsonify(data)


# 파이썬 명령어로 실행할 수 있음
if __name__ == '__main__':
    cf = joblib.load("./cf_factorization.pkl")
    mtrx = cf.get_complete_matrix()
    # mtrx.summary()
    app.run(host='127.0.0.1', port=8000, debug=True)