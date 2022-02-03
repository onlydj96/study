
# BayesianOptimization

from bayes_opt import BayesianOptimization

def black_box_funtion(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

pbounds = {'x' : (2, 4), 'y' : (-3, 3)}  # pbounds 안의 값은 범위를 설정하는 것이다. 예를 들어 x의 2, 4는 2부터 4의 실수범위를 나타낸다.



optimizer = BayesianOptimization(
    f = black_box_funtion,
    pbounds=pbounds,
    random_state=66
)

optimizer.maximize(
    init_points=2,   # 초기 랜덤 포인트 갯수
    n_iter=0         # 반복 횟수 (많을 수록 정확한 값을 얻을 수 있다)
)
