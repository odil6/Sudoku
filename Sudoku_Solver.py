import numpy as np


class Solver:

    def __init__(self, g):
        if g:
            self.grid = g
        else:
            self.grid = [[0, 3, 9, 1, 0, 0, 0, 0, 0],
                         [4, 0, 8, 0, 6, 0, 0, 0, 2],
                         [2, 0, 0, 5, 8, 0, 7, 0, 0],
                         [8, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 0, 0, 0, 9, 0, 0, 0],
                         [3, 0, 6, 0, 0, 0, 0, 4, 9],
                         [0, 0, 0, 0, 1, 0, 0, 3, 0],
                         [0, 4, 0, 3, 0, 0, 0, 0, 8],
                         [7, 0, 0, 0, 0, 0, 4, 0, 0]]
        self.solution = None
        self.solve()

    def get_grid(self):
        return self.grid

    def get_solution(self):
        return self.solution

    def set_solution(self, g):
        self.solution = g

    def possible(self, y, x, n):
        for i in range(0, 9):
            if self.grid[y][i] == n and x is not i:
                print('y, i', y, i)
                print('val = ', self.grid[y][i])

                print('a')
                return False
        for i in range(0, 9):
            if self.grid[i][x] == n and i is not y:
                return False
        x0 = (x // 3) * 3
        y0 = (y // 3) * 3
        for i in range(0, 3):
            for j in range(0, 3):
                if self.grid[y0 + i][x0 + j] == n and ((y0 + i) is not y and (x0 + j) is not x):
                    return False
        return True

    def get_empty(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] == 0:
                    return i, j  # row, col
        return None

    def solve(self):
        find = self.get_empty()
        if not find:
            return True
        else:
            y, x = find
        for n in range(1, 10):
            if self.possible(y, x, n):
                self.grid[y][x] = n
                if self.solve():
                    # self.solution = []
                    # for y in range(9):
                    #     self.solution.append([])
                    #     for x in range(9):
                    #         self.solution[y].append(self.grid[y][x])
                    self.grid[y][x] = 0
                    return True
                self.grid[y][x] = 0
        return False


