import numpy as np

xmax = 9
ymax = 9

centers = np.array([(1,1), ()]).astype(int)
grid = np.zeros((xmax,ymax)).astype(int)

def display(grid):
    for i in range(xmax):
        s = ""
        for j in range(ymax):
            s += str(grid[i][j]) + " "
        print(s)
    print()

for c in range(len(centers)):
    grid[centers[c][0], centers[c][1]] = c + 1

def get_opposite(x, y, color):
    dx = centers[color][0] - x
    dy = centers[color][1] - y
    return centers[color][0] + dx, centers[color][0] + dy

def mark(x, y, color, grid):
    if grid[x, y] == 0:
        c = get_opposite(x, y, color)
        if 0 <= c[0] < xmax and 0 <= c[1] < ymax and grid[c[0], c[1]] == 0:
            grid[x, y] = color + 1
            grid[c[0], c[1]] = color + 1
    return grid

def is_there_white():
    for i in range(xmax):
        for j in range(ymax):
            if grid[i, j] == 0:
                return True
    return False

def find_colors(x, y):
    colors = []
    if grid[x, y] == 0:
        for i in range(len(centers)):
            c = get_opposite(x, y, i)
            if 0 <= c[0] < xmax and 0 <= c[1] < ymax and grid[c[0], c[1]] == 0:
                colors.append(i)
    return colors

def iter(grid):
    for i in range(xmax):
        for j in range(ymax):
            if grid[i, j] == 0:
                colors = find_colors(i, j)
                if len(colors) == 1:
                    grid = mark(i, j, colors[0], grid)
    return grid

grid = iter(grid)
display(grid)