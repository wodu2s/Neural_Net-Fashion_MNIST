import math
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    """벡터의 크기 계산"""    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    """벡터 덧셈"""
    def add(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    """벡터 뺄셈"""
    def substract(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    """내적 메서드"""
    def dot(self, other):
        return self.x*other.x + self.y*other.y
    
    """
    str 객체
    : 객체를 print 했을 때 나오는 형태를 지정해주는 함수
    """
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
        
v1 = Vector2D(3,4)
v2 = Vector2D(1,2)

print(v1.magnitude())
v3 = v1.add(v2)
print(v3)

v4 = v1.substract(v2)
print(v4)

print(v1.dot(v2))