import theater_module
theater_module.price(3) # 3명이서 영화보러 갔을 때 가격
theater_module.price_morning(4)
theater_module.price_soldier(5)

# 모듈이름을 줄여서 불러오는 방법
# import theater_module as mv
# mv.price(3)
# mv.price_morning(4)
# mv.price_soldier(5)

# 더 줄이기
# from theater_module import * 
# price(3)
# price_morning(4)
# price_soldier(5)

from theater_module import price, price_morning # theater_module에서 price_morning 만 가져오기
from theater_module import price_soldier as price # price_soldier만 쓰기 때문에 price로 줄여서 사용
price(4)