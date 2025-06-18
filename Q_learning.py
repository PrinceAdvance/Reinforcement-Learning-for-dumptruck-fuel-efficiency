import numpy as np
import matplotlib.pyplot as plt
import random

# Grid constants
EMPTY, OBSTACLE, LOAD, DUMP, INCLINE = 0, 1, 2, 3, 4

# Actions: up, down, left, right
ACTIONS = ['U', 'D', 'L', 'R']
action_map = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

class MineEnv:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self._setup()
        self.reset()

    def _setup(self):
        """Initialize grid with obstacles, inclines, load & dump points"""
        self.grid[1, 1] = OBSTACLE
        self.grid[2, 2] = INCLINE
        self.load_pos = (0, 0)
        self.dump_pos = (4, 4)
        self.grid[self.load_pos] = LOAD
        self.grid[self.dump_pos] = DUMP

    def reset(self):
        """Reset truck to the load point"""
        self.pos = self.load_pos
        self.has_load = True  # Start with load to deliver
        return self._state()

    def _state(self):
        """Return current state as a tuple"""
        return (*self.pos, int(self.has_load))  # (x, y, load_status)

    def _in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def step(self, action):
        """Move the truck in the environment based on action"""
        dx, dy = action_map[action]
        x, y = self.pos
        new_x, new_y = x + dx, y + dy

        # Default penalty
        reward = -1
        done = False

        if not self._in_bounds(new_x, new_y) or self.grid[new_x, new_y] == OBSTACLE:
            # Hit wall or obstacle
            reward -= 5
            return self._state(), reward, done

        self.pos = (new_x, new_y)

        # Extra fuel penalty for inclines
        if self.grid[new_x, new_y] == INCLINE:
            reward -= 3

        # Delivering load
        if self.grid[new_x, new_y] == DUMP and self.has_load:
            reward += 20  # Successful delivery
            done = True
            self.has_load = False

        return self._state(), reward, done

    def render(self):
        """Display the grid with current truck position"""
        grid_disp = self.grid.copy().astype(str)
        grid_disp[self.pos] = 'T'
        print("\n".join(" ".join(row) for row in grid_disp))

