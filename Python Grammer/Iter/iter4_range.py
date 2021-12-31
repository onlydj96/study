
# range형 객체

'''
for - in 문에서 사용되는 range() 함수는 range형 객체를 만든다

range형 객체는 iter()함수를 통해 range_iterator 형으로 변환시킬 수 있는 iterator 객체이다

range_iterator형 객체는 next() 함수를 통해 다음 요소에 접근할 수 있다.
'''

r_iter = iter(range(5))
print(r_iter)   # <range_iterator object at 0x0000019C6E345BD0>

next(r_iter)  # 0 
next(r_iter)  # 1
next(r_iter)  # 2
next(r_iter)  # 3
print(next(r_iter))  # 4
