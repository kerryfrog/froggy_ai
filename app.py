from tqdm import tqdm_notebook as tqdm
import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask_restful import reqparse
import joblib
from . import famous_patterns

app = Flask(__name__)


class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        """
        :param R: rating matrix
        :param k: latent parameter
        :param learning_rate: alpha on weight update
        :param reg_param: beta on weight update
        :param epochs: training epochs
        :param verbose: print status
        """
        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose

    def fit(self):
        """
        training Matrix Factorization : Update matrix latent weight and bias

        참고: self._b에 대한 설명
        - global bias: input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
        - 정규화 기능. 최종 rating에 음수가 들어가는 것 대신 latent feature에 음수가 포함되도록 해줌.

        :return: training_process
        """

        # init latent features
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_items, self._k))

        # init biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):
            # rating이 존재하는 index를 기준으로 training
            xi, yi = self._R.nonzero()
            for i, j in zip(xi, yi):
                self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 5 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))

    def cost(self):
        """
        compute root mean square error
        :return: rmse cost
        """

        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        # 참고: http://codepractice.tistory.com/90
        xi, yi = self._R.nonzero()
        # predicted = self.get_complete_matrix()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - self.get_prediction(x, y), 2)
        return np.sqrt(cost/len(xi))

    def gradient(self, error, i, j):
        """
        gradient of latent feature for GD

        :param error: rating - prediction error
        :param i: user index
        :param j: item index
        :return: gradient of latent feature tuple
        """

        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq

    def gradient_descent(self, i, j, rating):
        """
        graident descent function

        :param i: user index of matrix
        :param j: item index of matrix
        :param rating: rating of (i,j)
        """

        # get error
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        # update biases
        self._b_P[i] += self._learning_rate * \
            (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * \
            (error - self._reg_param * self._b_Q[j])

        # update latent feature
        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq

    def get_prediction(self, i, j):
        """
        get predicted rating: user_i, item_j
        :return: prediction of r_ij
        """
        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)

    def get_complete_matrix(self):
        """
        computer complete matrix PXQ + P.bias + Q.bias + global bias

        - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)


def read_csv():
    mtrx = pd.read_csv("./sgd.csv", encoding='cp949')
    return mtrx


def append_mtrx(mtrx, new_user, new_pattern):
    data = {
        new_pattern: '5'  # '5' means score
    }
    df = pd.DataFrame(data, index=[new_user])
    mtrx = mtrx.append(df)
    return mtrx


def mf(mtrx):
    factorizer = MatrixFactorization(
        mtrx, k=30, learning_rate=0.05, reg_param=0.02, epochs=1, verbose=True)
    factorizer.fit()
    mtrx = factorizer.get_complete_matrix()
    mtrx = pd.DataFrame(mtrx)
    return mtrx


def named_mtrx(mtrx):
    mtrx = np.round(mtrx, 5)
    # mtrx의 column은 patternId, row는 userId로 바꾸기
    mtrx.columns = famous_patterns
    mtrx.columns.name = "patternId"
    mtrx.index.name = "userId"
    return mtrx


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
def get_recommend():
    try:
        parser = reqparse.RequestParser()
        parser.add_argument('targetId', type=str)
        parser.add_argument('userId', type=str)
        args = parser.parse_args()

        _targetId = args['targetId']
        _userId = args['userId']
        
        mtrx = read_csv()
        appended_mtrx = append_mtrx(mtrx, _userId, _targetId)
        mf_matrix = mf(appended_mtrx)
        result = named_mtrx(mf_matrix)

        return {'TargetId': args['targetId'], 'UserId': args['userId']}

    except Exception as e:
        app.logger.error(e)
        return {'error': str(e)}


# 파이썬 명령어로 실행할 수 있음
if __name__ == '__main__':
    # mtrx.summary()
    app.run(host='127.0.0.1', port=8000, debug=True)
