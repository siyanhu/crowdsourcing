import math

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, other):
        return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2)

    def __str__(self):
        return '('+str(self.x)+','+str(self.y)+')'

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)