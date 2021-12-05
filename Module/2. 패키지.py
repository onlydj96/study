# # import 로 모듈을 끌어오기
# import travel.thailand
# trip_to = travel.thailand.ThailandPackage()
# trip_to.detail()

# # import 구문에서는 바로 class 를 끌어올 수 없기 때문에 from - import 구문을 사용
# from travel.thailand import ThailandPackage
# trip_to = ThailandPackage()
# trip_to.detail()

# from travel import vietnam
# trip_to = vietnam.VietnamPackage()
# trip_to.detail()

# 라이브러리에서 직접 호출인지, 외부에서 호출하는건지에 따라...
# from travel import *
# trip_to = thailand.ThailandPackage()
# trip_to.detail()

import inspect
import random
print(inspect.getfile(random)) # ()안에 있는 파일이 어디있는지 찾기
# from travel import *
# print(inspect.getfile(thailand))