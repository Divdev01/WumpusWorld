class Coords():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        if (isinstance(other, Coords)):
            return self.x == other.x and self.y == other.y
    def __str__(self) -> str:
      return f'Coords({self.x},{self.y})'

    def __hash__(self):
        return hash((self.x,self.y))
  
