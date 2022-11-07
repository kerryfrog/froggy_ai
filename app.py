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
        'test': 'get data from flask server!',
        'som': 'dasom'
    }
    return jsonify(data)


@app.route('/newmodel', methods=['POST'])
def addRow():
    try:
        parser = reqparse.RequestParser()
        parser.add_argument('targetId', type=str)
        parser.add_argument('userId', type=str)
        args = parser.parse_args()

        _targetId = args['targetId']
        _userId = args['userId']
        return {'TargetId': args[_targetId], 'UserId': args['userId']}

    except Exception as e:
        return {'error': str(e)}


# 파이썬 명령어로 실행할 수 있음
if __name__ == '__main__':
    # mtrx.summary()
    app.run(host='127.0.0.1', port=8000, debug=True)
