import pygame
import sys

WINDOW_WIDTH, WINDOW_HEIGHT = 784, 784  # 28x28 grid, each cell is 28x28 pixels

window = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))

GRID_SIZE = 196
BOX_WIDTH, BOX_HEIGHT = WINDOW_WIDTH // GRID_SIZE , WINDOW_HEIGHT // GRID_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

LINE_WIDTH = 6  # Set the desired line width
RADIUS = LINE_WIDTH**2

grid = []

class Box:
    def __init__(self, i, j):
        self.x = i
        self.y = j
        self.wall = False
    
    def draw(self, win, color):
        pygame.draw.rect(win,color,(self.x*BOX_WIDTH,self.y*BOX_HEIGHT,BOX_WIDTH,BOX_HEIGHT))

for i in range(GRID_SIZE):
    arr = []
    for j in range(GRID_SIZE):
        arr.append(Box(i,j))
    grid.append(arr)


def index_in_grid(n):
    return (n-LINE_WIDTH >= 0) and (n+LINE_WIDTH <= GRID_SIZE)



def main():
    while True:
        for event in pygame.event.get():
            # Quit window
            if event.type == pygame.QUIT:
                pygame.quit() 
                sys.exit()
            # Click interaction:
            if event.type == pygame.MOUSEMOTION:
                x = pygame.mouse.get_pos()[0]
                y = pygame.mouse.get_pos()[1]
                if event.buttons[0]:
                    i = x // BOX_WIDTH
                    j = y // BOX_HEIGHT

                    # PAINT A CIRCLE:
                    for m in range(i-LINE_WIDTH, i + LINE_WIDTH):
                        for n in range(j-LINE_WIDTH,j+LINE_WIDTH):
                            if index_in_grid(m) and index_in_grid(n) and ((i-m)**2 + (j-n)**2 <= RADIUS):
                                grid[m][n].wall = True
        
        window.fill((0,0,0))

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                box = grid[i][j]
                box.draw(window,(255,255,255))
              
                if box.wall:
                    box.draw(window,(90,90,90))

            
        pygame.display.flip()

main()



    