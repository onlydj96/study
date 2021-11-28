# 마린 만들기

name = "마린"
hp = 40
damage = 5

print("{} 유닛이 생성되었습니다.".format(name))
print("체력 {0}, 공격력 {1}".format(hp, damage))

# 탱크 만들기
tank_name = "탱크"
tank_hp = 150
tank_damage = 35

print("{} 유닛이 생성되었습니다.".format(tank_name))
print("체력 {0}, 공격력 {1}".format(tank_hp, tank_damage))

def attack(name, location, damage):
    print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 {2}]".format(name, location, damage))

attack(name, "1시", damage) # 마린의 함수
attack(tank_name, "1시", tank_damage) # 탱크의 함수

# __init__ : self를 제외한 나머지 매게변수와 동이한 개수를 unit에 넣어주어야함

class Unit:
    def __init__(self, name, hp, damage):
        self.name = name
        self.hp = hp
        self.damage = damage
        print("{0} 유닛이 생성되었습니다.".format(self.name))
        print("체력 {0}, 공격력 {1}".format(self.hp, self.damage))
        
Marine1 = Unit("마린", 40, 5)
Marine2 = Unit("마린", 40, 5)
Tank = Unit("탱크", 150, 35)

print(Tank)