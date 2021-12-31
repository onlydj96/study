n = 100

n_iter = iter(n)

'''
Traceback (most recent call last):
  File "d:\Study\Iter\iter1.py", line 3, in <module>
    n_iter = iter(n)
TypeError: 'int' object is not iterable

* 정수형 변수는 iterable하지 않다. 
  - 따라서 iter()함수를 통해 iterator 객체로 바꿀 수 없다.
  - TypeError 예외를 발생시킨다.
'''

