import pygame
import math
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("El Asterisco")

# Colors
COLORS = {
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 255, 0),
    "YELLOW": (255, 255, 0),
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "PURPLE": (128, 0, 128),
    "ORANGE": (255, 165, 0),
    "GREY": (128, 128, 128),
    "TURQUOISE": (64, 224, 208),
}

class Node:
    def __init__(self, row, col, size, grid_size):
        self.row = row
        self.col = col
        self.x = row * size
        self.y = col * size
        self.size = size
        self.grid_size = grid_size
        self.color = COLORS["WHITE"]
        self.neighbors = []

    def get_position(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == COLORS["RED"]

    def is_open(self):
        return self.color == COLORS["GREEN"]

    def is_barrier(self):
        return self.color == COLORS["BLACK"]

    def is_start(self):
        return self.color == COLORS["ORANGE"]

    def is_end(self):
        return self.color == COLORS["TURQUOISE"]

    def reset(self):
        self.color = COLORS["WHITE"]

    def set_start(self):
        self.color = COLORS["ORANGE"]

    def set_closed(self):
        self.color = COLORS["RED"]

    def set_open(self):
        self.color = COLORS["GREEN"]

    def set_barrier(self):
        self.color = COLORS["BLACK"]

    def set_end(self):
        self.color = COLORS["TURQUOISE"]

    def set_path(self):
        self.color = COLORS["PURPLE"]

    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.size, self.size))

    def update_neighbors(self, grid):
        self.neighbors = []
        directions = [  # including diagonal directions
            (1, 0),  # Down
            (-1, 0),  # Up
            (0, 1),  # Right
            (0, -1),  # Left
            (1, 1),  # Down-Right
            (1, -1),  # Down-Left
            (-1, 1),  # Up-Right
            (-1, -1),  # Up-Left
        ]

        for d_row, d_col in directions:
            new_row, new_col = self.row + d_row, self.col + d_col
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                if abs(d_row) + abs(d_col) == 2:  # Diagonal move
                    # Check if both adjacent cells are not barriers
                    if (
                        not grid[self.row + d_row][self.col].is_barrier()
                        and not grid[self.row][self.col + d_col].is_barrier()
                    ):
                        if not grid[new_row][new_col].is_barrier():
                            self.neighbors.append(grid[new_row][new_col])
                else:
                    if not grid[new_row][new_col].is_barrier():
                        self.neighbors.append(grid[new_row][new_col])

    def __lt__(self, other):
        return False

def heuristic(start, end):
    x1, y1 = start
    x2, y2 = end
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw_fn):
    while current in came_from:
        current = came_from[current]
        current.set_path()
        draw_fn()

import pygame
import math
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("El Asterisco")

# Colors
COLORS = {
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 255, 0),
    "YELLOW": (255, 255, 0),
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "PURPLE": (128, 0, 128),
    "ORANGE": (255, 165, 0),
    "GREY": (128, 128, 128),
    "TURQUOISE": (64, 224, 208),
}

class Node:
    def __init__(self, row, col, size, grid_size):
        self.row = row
        self.col = col
        self.x = row * size
        self.y = col * size
        self.size = size
        self.grid_size = grid_size
        self.color = COLORS["WHITE"]
        self.neighbors = []

    def get_position(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == COLORS["RED"]

    def is_open(self):
        return self.color == COLORS["GREEN"]

    def is_barrier(self):
        return self.color == COLORS["BLACK"]

    def is_start(self):
        return self.color == COLORS["ORANGE"]

    def is_end(self):
        return self.color == COLORS["TURQUOISE"]

    def reset(self):
        self.color = COLORS["WHITE"]

    def set_start(self):
        self.color = COLORS["ORANGE"]

    def set_closed(self):
        self.color = COLORS["RED"]

    def set_open(self):
        self.color = COLORS["GREEN"]

    def set_barrier(self):
        self.color = COLORS["BLACK"]

    def set_end(self):
        self.color = COLORS["TURQUOISE"]

    def set_path(self):
        self.color = COLORS["PURPLE"]

    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.size, self.size))

    def update_neighbors(self, grid):
        self.neighbors = []
        directions = [  # including diagonal directions
            (1, 0),  # Down
            (-1, 0),  # Up
            (0, 1),  # Right
            (0, -1),  # Left
            (1, 1),  # Down-Right
            (1, -1),  # Down-Left
            (-1, 1),  # Up-Right
            (-1, -1),  # Up-Left
        ]

        for d_row, d_col in directions:
            new_row, new_col = self.row + d_row, self.col + d_col
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                if abs(d_row) + abs(d_col) == 2:  # Diagonal move
                    # Check if both adjacent cells are not barriers
                    if (
                        not grid[self.row + d_row][self.col].is_barrier()
                        and not grid[self.row][self.col + d_col].is_barrier()
                    ):
                        if not grid[new_row][new_col].is_barrier():
                            self.neighbors.append(grid[new_row][new_col])
                else:
                    if not grid[new_row][new_col].is_barrier():
                        self.neighbors.append(grid[new_row][new_col])

    def __lt__(self, other):
        return False

def heuristic(start, end):
    x1, y1 = start
    x2, y2 = end
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw_fn):
    while current in came_from:
        current = came_from[current]
        current.set_path()
        draw_fn()

def a_star_algorithm(draw_fn, grid, start, end):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = heuristic(start.get_position(), end.get_position())
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[1]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw_fn)
            end.set_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor.get_position(), end.get_position())

                if neighbor not in open_set_hash:
                    open_set.put((f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.set_open()

        draw_fn()

        if current != start:
            current.set_closed()

    return False

def create_grid(grid_size, width):
    grid = []
    node_size = width // grid_size
    for i in range(grid_size):
        grid.append([Node(i, j, node_size, grid_size) for j in range(grid_size)])
    return grid

def draw_grid(window, grid_size, width):
    gap = width // grid_size
    for i in range(grid_size):
        pygame.draw.line(window, COLORS["GREY"], (0, i * gap), (width, i * gap))
        for j in range(grid_size):
            pygame.draw.line(window, COLORS["GREY"], (j * gap, 0), (j * gap, width))

def draw(window, grid, grid_size, width):
    window.fill(COLORS["WHITE"])
    for row in grid:
        for node in row:
            node.draw(window)
    draw_grid(window, grid_size, width)
    pygame.display.update()

def get_clicked_position(pos, grid_size, width):
    gap = width // grid_size
    y, x = pos
    return y // gap, x // gap

def main(window, width):
    grid_size = 9
    grid = create_grid(grid_size, width)
    start = None
    end = None

    running = True
    while running:
        draw(window, grid, grid_size, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if pygame.mouse.get_pressed()[0]:  # Left-click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, grid_size, width)
                node = grid[row][col]
                if not start and node != end:
                    start = node
                    start.set_start()

                elif not end and node != start:
                    end = node
                    end.set_end()

                elif node != end and node != start:
                    node.set_barrier()

            elif pygame.mouse.get_pressed()[2]:  # Right-click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, grid_size, width)
                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                elif node == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)

                    a_star_algorithm(lambda: draw(window, grid, grid_size, width), grid, start, end)

                if event.key == pygame.K_c:  # Clear
                    start = None
                    end = None
                    grid = create_grid(grid_size, width)

    pygame.quit()

main(WIN, WIDTH)


def create_grid(grid_size, width):
    grid = []
    node_size = width // grid_size
    for i in range(grid_size):
        grid.append([Node(i, j, node_size, grid_size) for j in range(grid_size)])
    return grid

def draw_grid(window, grid_size, width):
    gap = width // grid_size
    for i in range(grid_size):
        pygame.draw.line(window, COLORS["GREY"], (0, i * gap), (width, i * gap))
        for j in range(grid_size):
            pygame.draw.line(window, COLORS["GREY"], (j * gap, 0), (j * gap, width))

def draw(window, grid, grid_size, width):
    window.fill(COLORS["WHITE"])
    for row in grid:
        for node in row:
            node.draw(window)
    draw_grid(window, grid_size, width)
    pygame.display.update()

def get_clicked_position(pos, grid_size, width):
    gap = width // grid_size
    y, x = pos
    return y // gap, x // gap

def main(window, width):
    grid_size = 14
    grid = create_grid(grid_size, width)
    start = None
    end = None

    running = True
    while running:
        draw(window, grid, grid_size, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if pygame.mouse.get_pressed()[0]:  # Left-click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, grid_size, width)
                node = grid[row][col]
                if not start and node != end:
                    start = node
                    start.set_start()

                elif not end and node != start:
                    end = node
                    end.set_end()

                elif node != end and node != start:
                    node.set_barrier()

            elif pygame.mouse.get_pressed()[2]:  # Right-click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, grid_size, width)
                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                elif node == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)

                    a_star_algorithm(lambda: draw(window, grid, grid_size, width), grid, start, end)

                if event.key == pygame.K_c:  # Clear
                    start = None
                    end = None
                    grid = create_grid(grid_size, width)

    pygame.quit()

main(WIN, WIDTH)
