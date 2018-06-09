import numpy as np
import random
import itertools
import os
from math import sqrt

from dijk import get_dist_map

class Game:
    def __init__(self, track, width, height, show_game=True):
        self.track = track
        self.width = len(track[0]) - 1
        self.height = len(track)
        self.show_game = show_game

        self.x = None
        self.y = None
        self.dx = None
        self.dy = None

        self.ostate = np.zeros((self.width, self.height))
        self.tstate = np.zeros((self.width, self.height))
        for y in range(self.height):
            for x in range(self.width):
                c = self.track[y][x]
                if c == "@":
                    self.sx = x
                    self.sy = y
                elif c == "#":
                    self.ostate[x, y] = 1
                elif c == "*":
                    self.tx = x
                    self.ty = y
                    self.tstate[x, y] = 1
        self.dist_map = get_dist_map(width, height, self.ostate, self.tx, self.ty)

        self.steps = 0
        self.total_reward = 0.
        self.current_reward = 0.
        self.total_game = 0
        self.total_success = 0

    def _get_state(self):
        cstate = np.zeros((self.width, self.height))
        nstate = np.zeros((self.width, self.height))

        if 0 <= self.x and self.x < self.width and 0 <= self.y and self.y < self.height:
            cstate[self.x, self.y] = 1

        nx = self.x + self.dx
        ny = self.y + self.dy
        if 0 <= nx and nx < self.width and 0 <= ny and ny < self.height:
            nstate[nx, ny] = 1

        r = np.append(cstate, nstate, axis=0)
        r = np.append(r, self.ostate, axis=0)
        r = np.append(r, self.tstate, axis=0)
        return r

    def _draw_screen(self):
        os.system("clear")

        title = str(self.total_success) + " SUCCESSES / " + str(self.total_game) + " GAMES"
        print(title)

        for y in range(self.height):
            for x in range(self.width):
                if self.x == x and self.y == y:
                    print("o", end="")
                elif self.tstate[x, y]:
                    print("O", end="")
                elif self.ostate[x, y]:
                    print("X", end="")
                else:
                    print("_", end="")
            print()

    def reset(self):
        self.steps = 0
        self.current_reward = 0
        self.total_game += 1

        self.x = self.sx
        self.y = self.sy
        self.dx = 0
        self.dy = 0

        return self._get_state()

    def _update_velocity(self, action):
        # 0: up_left, 1: left, 2: down_left, 3: up, 4: nop, 5: down, 6: up_right, 7: right, 8: down_right
        self.dx += int(action / 3) - 1
        self.dy += action % 3 - 1

    def _update_car(self):
        prd = self.dist_map[(self.x, self.y)]
        self.x += self.dx
        self.y += self.dy
        rd = self.dist_map[(self.x, self.y)]
        return prd - rd

    def _is_gameover_move(self):
        d = sqrt(self.dx * self.dx + self.dy * self.dy)
        x = self.x
        y = self.y
        for i in range(int(d)):
            x += self.dx / d
            y += self.dy / d
            if self._is_illegal(int(x), int(y)):
                return True
        return self._is_illegal(self.x + self.dx, self.y + self.dy)

    def _is_illegal(self, x, y):
        return x < 0 or x >= self.width or \
            y < 0 or y >= self.height or \
            self.ostate[x, y]

    def _will_gameover(self, after):
        self.x += self.dx * (after - 1)
        self.y += self.dy * (after - 1)
        b = self._is_gameover_move()
        self.x -= self.dx * (after - 1)
        self.y -= self.dy * (after - 1)
        return b

    def _is_success(self):
        return self.x >= 0 and self.x < self.width and \
            self.y >= 0 and self.y < self.height and \
            self.tstate[self.x, self.y]

    def step(self, action):
        self.steps += 1

        self._update_velocity(action)
        gameover = self._is_gameover_move()

        if gameover:
            reward = -(self.height + self.width)
        else:
            reward = self._update_car() - 1
            success = self._is_success()
            if success:
                reward = (self.height + self.width)
                self.total_success += 1
            elif self._will_gameover(1):
#                reward -= (self.height + self.width) / 5
                reward -= 3
            elif self._will_gameover(2):
#                reward -= (self.height + self.width) / 10
                reward -= 2
            elif self._will_gameover(3):
#                reward -= (self.height + self.width) / 15
                reward -= 1

        self.current_reward += reward

        timeout = self.steps > 10000
        if timeout:
            print("TIMEOUT")

        end = gameover or success or timeout
        if end:
            self.total_reward += self.current_reward

        if self.show_game:
            self._draw_screen()

        return self._get_state(), reward, end
