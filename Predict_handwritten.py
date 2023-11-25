import pygame
import sys
import numpy as np
import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay

# Import the datasets:
training = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")


print(training.head())
print(training.describe())

Y_train = training["label"]
X_train = training.iloc[:,1:]

Y_test = test["label"]
X_test = test.iloc[:,1:]

# Train the model:
model4 = SVC(kernel="poly",degree=4) # OPTIMAL
model4.fit(X_train,Y_train)
print(f"Model C=4 score: {model4.score(X_test,Y_test)}")



while input("Testekin amaitu nahi al duzu? (0 bukatzeko)") != "0":
    # Initialize Pygame
    pygame.init()

    # Constants
    WIDTH, HEIGHT = 784, 784  # 28x28 grid, each cell is 28x28 pixels
    GRID_SIZE = 28
    GRID_WIDTH = WIDTH // GRID_SIZE
    GRID_HEIGHT = HEIGHT // GRID_SIZE
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    LINE_WIDTH = 2  # Set the desired line width



    # Create a 2D grid to store the state of each cell
    grid = [[WHITE for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    # Create the Pygame window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Draw on Grid")

    # Main loop
    drawing = False
    running = True
    flattened_grid = None  # Variable to store the flattened grid
    while running:
        for event in pygame.event.get():

            if event.type == pygame.QUIT: # If the window is closed:
                running = False # Stop the loop
                flattened_grid = [254 if cell == BLACK else 0 for row in grid for cell in row] # Save the image

            elif event.type == pygame.MOUSEBUTTONDOWN: # If leftclick is pressed
                drawing = True

            elif event.type == pygame.MOUSEBUTTONUP: # If leftclick is released
                drawing = False

            elif event.type == pygame.MOUSEMOTION and drawing: # If we are leftclicking and moving the mouse
                # Get the position of the mouse
                x, y = event.pos
                # Map the mouse position to the grid cell
                grid_x = x // GRID_SIZE
                grid_y = y // GRID_SIZE
                # Update the color of the cell to black
                grid[grid_y][grid_x] = BLACK

                # Draw a rectangle to make the line thicker
                for i in range(LINE_WIDTH):
                    for j in range(LINE_WIDTH):
                        if 0 <= grid_x + i < GRID_WIDTH and 0 <= grid_y + j < GRID_HEIGHT:
                            grid[grid_y + j][grid_x + i] = BLACK

        # Draw the grid
        for y, row in enumerate(grid):
            for x, color in enumerate(row):
                pygame.draw.rect(screen, color, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Update the display
        pygame.display.flip()
    
    # Print the flattened grid after the window is closed
    x = np.array([flattened_grid])
    for i in range(784):
        print(x[0,i],sep="    ",end="")
        if (i+1) % 28 == 0:
            print()
    print("--------------------------")
    print(f"Modeloaren predikzioa: {model4.predict(x)} da")
    print("--------------------------")
    pygame.quit()

   
 # Quit Pygame
pygame.quit()
sys.exit()
#print(f"PREDICTION: {model4.predict(training.iloc[1,:].reshape(-1,1))}")
