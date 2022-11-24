from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from flask import Flask, request
from famous_patterns import get_patterns


app = Flask(__name__)

patterns = get_patterns()


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
        try:
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

        except Exception as e:
            app.logger.error(e)
            print(e)
            return {'error': str(e)}

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


def get_rmse(R, P, Q, non_zeros):
    error = 0
    # 두개의 분해된 행렬 P와 Q.T의 내적 곱으로 예측 R 행렬 생성
    full_pred_matrix = np.dot(P, Q.T)

    # 실제 R 행렬에서 널이 아닌 값의 위치 인덱스 추출하여 실제 R 행렬과 예측 행렬의 RMSE 추출
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]

    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind,
                                                  y_non_zero_ind]

    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)

    return rmse


def matrix_factorization(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape
    # P와 Q 매트릭스의 크기를 지정하고 정규분포를 가진 랜덤한 값으로 입력합니다.
    np.random.seed(1)
    P = np.random.normal(scale=1./K, size=(num_users, K))
    Q = np.random.normal(scale=1./K, size=(num_items, K))

    break_count = 0
    # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트 객체에 저장.
    non_zeros = [(i, j, R[i, j]) for i in range(num_users)
                 for j in range(num_items) if R[i, j] > 0]

    # SGD기법으로 P와 Q 매트릭스를 계속 업데이트.
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구함
            eij = r - np.dot(P[i, :], Q[j, :].T)
            # Regularization을 반영한 SGD 업데이트 공식 적용
            P[i, :] = P[i, :] + learning_rate * \
                (eij * Q[j, :] - r_lambda*P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * \
                (eij * P[i, :] - r_lambda*Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)
        if step % 10 == 0:
            print("### iteration step : ", step, " rmse : ", np.round(rmse, 7))

    return P, Q


def predict(mtrx):
    mtrx = pd.DataFrame(mtrx)
    # mtrx['pattern_id'] = pd.to_numeric(mtrx['pattern_id'])
    P, Q = matrix_factorization(
        mtrx.values, K=30, steps=50, learning_rate=0.01, r_lambda=0.01)
    pred_matrix = np.dot(P, Q.T)  # P @ Q.T 도 가능
    print('실제 행렬:\n', mtrx)
    print('\n예측 행렬:\n', np.round(pred_matrix, 2))
    return pred_matrix


def read_csv():
    mtrx = pd.read_csv("./sgd.csv", encoding='cp949', index_col=0)
    return mtrx


def append_mtrx(mtrx, new_user, score_list):
    data = {} #
    for score in score_list:
        _patternId = score['id']
        _score = score['score']
        _patternId = str(_patternId)

        data[_patternId] = _score

    df = pd.DataFrame(data, index=[new_user])
    mtrx = pd.concat([mtrx, df])

    # mtrx.columns = [patterns.index]
    mtrx = mtrx.fillna(0)
    print(mtrx)
    

    return mtrx


def mf(mtrx):
    mtrx = mtrx.to_numpy()
    factorizer = MatrixFactorization(
        mtrx, k=30, learning_rate=0.05, reg_param=0.02, epochs=1, verbose=True)
    factorizer.fit()
    mtrx = factorizer.get_complete_matrix()
    mtrx = pd.DataFrame(mtrx)
    print(mtrx)
    return mtrx


def name_mtrx(mtrx):
    mtrx = np.round(mtrx, 5)
    # mtrx의 column은 patternId, row는 userId로 바꾸기

    mtrx.columns = patterns
    mtrx.columns.name = "patternId"
    mtrx.index.name = "userId"
    return mtrx


def get_max(mtrx, user, index):
    cur_user_index = 5622 + index
    new_row = mtrx[mtrx.index == cur_user_index]

    # 최댓값의 인덱스 찾는 코드
    maxValueIndex = new_row.idxmax(axis=1)
    result = maxValueIndex[cur_user_index]

    # 최댓값 n개 찾기

    top_n = 10
    # new_row = new_row.argsort()[::-1]
    # rank = new_row.flatten()
    # rank = rank.tolist()

    # print(rank[top_n])
    # print(result)
    # print(mtrx.sort_values(by=cur_user_index, axis=1))
    # print(type(new_row))
    predicted_scores = new_row.values.tolist()
    # print(predicted_scores)
    return predicted_scores


@app.route('/recommend', methods=['POST'])
def parse():
    try:
        # parser = reqparse.RequestParser()
        # parser.add_argument('userScoreList', type=list)
        # args = parser.parse_args()
        # _userScoreList= args['userScoreList']

        userScoreList = request.get_json(force=True)
        userScoreList = userScoreList['userScoreList']
        recommendList = []

        mtrx = read_csv()
        for i in range(len(userScoreList)):
            userScore = userScoreList[i]
            _userId = userScore['user']
            scoreList = userScore['scoreList']

            for eachScore in scoreList:
                _id = eachScore['id']
                _score = eachScore['score']
                # print(_id, _score)
            print("mtrx",mtrx)
            mtrx = pd.DataFrame(mtrx)
            
            appended_mtrx = append_mtrx(mtrx, _userId, scoreList) #
            print("append",appended_mtrx)
            mtrx = appended_mtrx

            mf_matrix = mf(appended_mtrx)
            named = name_mtrx(mf_matrix)
            result = get_max(named, _userId, i)
            recommendList.append((_userId, result))
        return recommendList

    except Exception as e:
        app.logger.error(e)
        return {'error': str(e)}
    return 0


# 파이썬 명령어로 실행할 수 있음
if __name__ == '__main__':
    # mtrx.summary()
    app.run(host='127.0.0.1', port=8000, debug=True)
