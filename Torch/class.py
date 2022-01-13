
'''
1. Class 
  
 변수와 함수를 묶어서 하나의 새로운 객체(타입)을 만드는 것 

2. self

 클래스를 저장할 변수를 의미한다. class가 변수로 지정되었을 때 그 변수를 통해서 여러 내장함수를 불러올 수 있다.

3. __init__

클래스를 불러올 때 가지는 디폴트 값을 의미한다. 즉 __init__은 클래스의 생성자 역할을 하고 있으며

4. __call__

함수를 호출 하는 것처럼 클래스의 객체도 호출하게 만들어주는 메서드가 __call__이다.  
즉, __init__ 은 인스턴스 초기화를 위해, __call__ 은 인스턴스가 호출됐을 때 실행된다

5. 상속

 상속은 새로 만드는 class()에 전에 만든 클래스를 넣을 수 있는 기능으로서 만든 클래스를 새로 만든 클래스로 불러와 그 안의 기능들을 사용할 수 있다.
__super__ 기능을 통해서 상속된 클래스의 함수들또한 불러올 수 있다.
'''


class JSS:
    def __init__(self):
        self.name = input('이름 : ')
        self.age = input('나이 : ')
    def show(self):
        print("나의 이름은 {1}, 나이는 {2}세 입니다.".format(self.name, self.age))


a = JSS()   # a라는 변수로 클래스 JSS를 지정해준다.
print(a.name)
a.show()  # a라는 변수에 JSS라는 클래스를 지정해주었기 때문에 JSS의 내장함수인 show를 불러올 수 있다.


class JSS2(JSS):            # 클래스를 상속시키기 위해서는 새롭게 정의되는 클래스 괄호안에 상속시킬 클래스를 넣어주면 된다.
    def __init__(self):
        super().__init__()          # 상속된 클래스의 __init__함수를 불러온다.
        self.gender = input("성별 : ")
    def show(self):
        print("나의 이름은 {1}, 성별은 {2}자, 나이는 {3}세 입니다.".format(self.name, self.gender, self.age))


class Calc:
    def __init__(self, n1, n2):
        self.n1 = n1 
        self.n2 = n2
        return print(self.n1, self.n2)

    def __call__(self, n1, n2):
        self.n1 = n1 
        self.n2 = n2
        return print(self.n1 + self.n2)

s = Calc(1,2)

s(7,8)         # call 기능을 통해 s라는 변수(인스턴스)를 통해 바로 클래스의 함수가 호출된다.