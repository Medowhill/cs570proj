import random

class GameObj:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.dr = 0
        self.dc = 0

    def _move(self, w, h, objs):
        nr = self.row + self.dr
        nc = self.col + self.dc
        pos = GameObj(nr, nc)

        if 0 <= nr and nr < h and 0 <= nc and nc < w and \
          not list(filter(lambda x: pos.same_pos(x), objs)):
            self.row = nr
            self.col = nc
        else:
            self.reverse()

    def reverse(self):
        self.dr *= -1
        self.dc *= -1

    def same_pos(self, obj):
        return self.row == obj.row and self.col == obj.col

class Car(GameObj):
    def __init__(self):
        GameObj.__init__(self, 0, 0)
        self.max = 0

    def move(self, action, w, h):
        if action == 1: # up
            self.dr = -1
            self.dc = 0
        elif action == 2: # right
            self.dr = 0
            self.dc = 1
        elif action == 3: # down
            self.dr = 1
            self.dc = 0
        elif action == 4: # left
            self.dr = 0
            self.dc = -1
        else: # nop
            self.dr = 0
            self.dc = 0

        pr = self.row
        pc = self.col
        self._move(w, h, [])

#        dist = self.row + self.col
#        if self.max < dist:
#            self.max = dist
#            return 1
#        elif self.max - (w + h) / 10 >= dist:
#            return -1
#        else:
#            return 0
        if pr < self.row or pc < self.col:
            return 1
        elif pr == self.row and pc == self.col:
            return -1
        else:
            return -2

class Target(GameObj):
    def __init__(self, w, h):
        GameObj.__init__(self, h - 1, w - 1)

class Bunker(GameObj):
    pass

class Obstacle(GameObj):
    def __init__(self, row, col):
        GameObj.__init__(self, row, col)

        r = random.randrange(2)
        v = random.randrange(2) * 2 - 1
        if r:
            self.dr = v
            self.dc = 0
        else:
            self.dr = 0
            self.dc = v

    def move(self, w, h, objs):
        self._move(w, h, objs)
