import numpy as np
import random
import itertools
import os

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

        self.sx = None
        self.sy = None
        self.tx = None
        self.ty = None

        self.ostate = None
        self.tstate = None

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

        if self._is_gameover():
            print("GAME OVER!!")
        elif self._is_success():
            print("SUCCESS!!")

    def reset(self):
        self.steps = 0
        self.current_reward = 0
        self.total_game += 1

        self.dx = 0
        self.dy = 0

        self.ostate = np.zeros((self.width, self.height))
        self.tstate = np.zeros((self.width, self.height))
        for y in range(self.height):
            for x in range(self.width):
                c = self.track[y][x]
                if c == "@":
                    self.x = x
                    self.y = y
                    self.sx = x
                    self.sy = y
                elif c == "#":
                    self.ostate[x, y] = 1
                elif c == "*":
                    self.tx = x
                    self.ty = y
                    self.tstate[x, y] = 1

        return self._get_state()

    def _update_car(self, action):
        # 0: up_left, 1: left, 2: down_left, 3: up, 4: nop, 5: down, 6: up_right, 7: right, 8: down_right
        self.dx += int(action / 3) - 1
        self.dy += action % 3 - 1

        p_from_start = abs(self.x - self.sx) + abs(self.y - self.sy)
        p_to_target = abs(self.x - self.tx) + abs(self.y - self.ty)

        self.x += self.dx
        self.y += self.dy

        from_start = abs(self.x - self.sx) + abs(self.y - self.sy)
        to_target = abs(self.x - self.tx) + abs(self.y - self.ty)

        reward = 0
        if p_from_start < from_start:
            reward += 1
        if p_to_target > to_target:
            reward += 1
        return reward

    def _is_gameover(self):
        return self.x < 0 or self.x >= self.width or \
            self.y < 0 or self.y >= self.height or \
            self.ostate[self.x, self.y]

    def _will_gameover(self, after):
        self.x += self.dx * after
        self.y += self.dy * after
        b = self._is_gameover()
        self.x -= self.dx * after
        self.y -= self.dy * after
        return b

    def _is_success(self):
        return self.x >= 0 and self.x < self.width and \
            self.y >= 0 and self.y < self.height and \
            self.tstate[self.x, self.y]

    def step(self, action):
        self.steps += 1

        reward = self._update_car(action)

        if self._will_gameover(1):
            reward -= 2

        gameover = self._is_gameover()
        success = self._is_success()

        if gameover:
            reward = -(self.height + self.width)
        elif success:
            reward = (self.height + self.width)
            self.total_success += 1
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
