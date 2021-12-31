l = [2, 4, 6, 8]
print(type(l))  # <class 'list'>

l_iter = iter(l)
print(type(l_iter))  # <class 'list_iterator'>

print(next(l_iter))  # 2
print(next(l_iter))  # 4
print(next(l_iter))  # 6
print(next(l_iter))  # 8

# next() 함수를 이용해서 iterator 객체의 다음 요소(element)를 얻는다


a_iter = iter(l)
a_iter.__next__()  #  2
a_iter.__next__()  #  4 
a_iter.__next__()  #  6
print(a_iter.__next__())  # 8

# __next__() 메소드를 이용해서 iterator 객체의 다음 요소(element)를 얻을 수 있다.

print(a_iter.__next__())  # 단 더 이상 가져올 객체가 없을 경우 StopIteration 예외를 발생시킨다.
