import numpy as np
import random
import itertools
import os

from objects import GameObj, Car, Obstacle, Bunker, Target

class Game:
    def __init__(self, width, height, obs, bun, show_game=True):
        self.width = width
        self.height = height
        self.show_game = show_game
        self.num_of_obstacles = obs
        self.num_of_bunkers = bun

        self.car = None
        self.target = Target(width, height)
        self.obstacles = None
        self.bunkers = None

        self.total_reward = 0.
        self.current_reward = 0.
        self.total_game = 0
        self.total_success = 0

    def _get_state(self):
        cstate = np.zeros((self.width, self.height))
        vstate = np.zeros((self.width, self.height))
        hstate = np.zeros((self.width, self.height))
        bstate = np.zeros((self.width, self.height))
#        mstate = np.zeros((self.width, self.height))

        cstate[self.car.row, self.car.col] = 1
        for o in self.obstacles:
            if o.dr != 0:
                vstate[o.row, o.col] = 1
            else:
                hstate[o.row, o.col] = 1
        for b in self.bunkers:
            bstate[b.row, b.col] = 1
#        for r in range(self.height):
#            for c in range(self.width):
#                if r + c <= self.car.max:
#                    mstate[r, c] = 1
#                else:
#                    break

        r = np.append(cstate, vstate, axis=0)
        r = np.append(r, hstate, axis=0)
        r = np.append(r, bstate, axis=0)
#        r = np.append(r, mstate, axis=0)
        return r

    def _draw_screen(self):
        os.system("clear")

        title = str(self.total_success) + " SUCCESSES / " + str(self.total_game) + " GAMES"
        print(title)

        for r in range(self.height):
            for c in range(self.width):
                pos = GameObj(r, c)

                if self.car.row == r and self.car.col == c:
                    print("o", end="")
                elif self.target.row == r and self.target.col == c:
                    print("O", end="")
                elif list(filter(lambda o: o.same_pos(pos), self.obstacles)):
                    print("X", end="")
                elif list(filter(lambda b: b.same_pos(pos), self.bunkers)):
                    print("B", end="")
                else:
                    print("_", end="")
            print()

        if self._is_gameover():
            print("GAME OVER!!")
        elif self._is_success():
            print("SUCCESS!!")

    def reset(self):
        self.steps = 0
        self.current_reward = 0
        self.total_game += 1

        self.car = Car()
        self.obstacles = []
        self.bunkers = []
        
        l = list(itertools.product(range(self.height), range(self.width)))
        random.shuffle(l)
        i = 0
        n = 0
        while n < self.num_of_obstacles + self.num_of_bunkers:
            r, c = l[i]
            if (r == 0 and c == 0) or (r == self.height - 1 and c == self.width - 1):
                pass
            else:
                if n < self.num_of_obstacles:
                    self.obstacles.append(Obstacle(r, c))
                else:
                    self.bunkers.append(Bunker(r, c))
                n += 1
            i += 1

        return self._get_state()

    def _update_car(self, action):
        return self.car.move(action, self.width, self.height)

    def _update_obstacles(self):
        objs = self.bunkers + [self.target]
        for o in self.obstacles:
            o.move(self.width, self.height, objs)

    def _is_gameover(self):
        return list(filter(lambda o: o.same_pos(self.car), self.obstacles))

    def _is_success(self):
        return self.car.same_pos(self.target)

    def step(self, action):
        move_reward = self._update_car(action)
        gameover = self._is_gameover()
        success = self._is_success()

        if (not gameover) and (not success):
            self._update_obstacles()
        gameover = self._is_gameover()

        if gameover:
            reward = -(self.height + self.width)
        elif success:
            self.total_success += 1
            reward = (self.height + self.width)
        else:
            reward = move_reward
        self.current_reward += reward

        end = gameover or success
        if end:
            self.total_reward += self.current_reward

        if self.show_game:
            self._draw_screen()

        return self._get_state(), reward, end, success
