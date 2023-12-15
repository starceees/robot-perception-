import pygame
import sys

# Initialize pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Navigation")

# Define colors
WHITE = (255, 255, 255)

# Robot position and movement
robot_x, robot_y = 50, 50
robot_speed = 5

# Maze representation (you'll need to adapt this to your specific maze)
maze = [
    "#########",
    "#       #",
    "#   ### #",
    "#   #   #",
    "#   #   #",
    "#   ### #",
    "#       #",
    "#########",
]

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Move the robot based on key presses
    if keys[pygame.K_UP]:
        new_x, new_y = robot_x, robot_y - robot_speed
    elif keys[pygame.K_DOWN]:
        new_x, new_y = robot_x, robot_y + robot_speed
    elif keys[pygame.K_LEFT]:
        new_x, new_y = robot_x - robot_speed, robot_y
    elif keys[pygame.K_RIGHT]:
        new_x, new_y = robot_x + robot_speed, robot_y
    else:
        continue

    # Check for collisions with maze walls
    row = new_y // 50
    col = new_x // 50

    if 0 <= row < len(maze) and 0 <= col < len(maze[0]) and maze[row][col] == " ":
        robot_x, robot_y = new_x, new_y

    # Clear the screen
    screen.fill(WHITE)

    # Draw the maze
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == "#":
                pygame.draw.rect(screen, (0, 0, 0), (x * 50, y * 50, 50, 50))

    # Draw the robot
    pygame.draw.rect(screen, (0, 128, 0), (robot_x, robot_y, 50, 50))

    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()
