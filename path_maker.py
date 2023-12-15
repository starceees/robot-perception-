import numpy as np
import cv2
import math

# Define initial orientation as a direction vector (pointing upwards)
orientation = np.array([0, -1])  # Upwards

def update_orientation(current_orientation, command):
    angle_change = 2.5
    # Each command changes the orientation by 10 degrees

    if 'Action.RIGHT' in command:
        angle = np.deg2rad(angle_change)  # Convert to radians
    elif 'Action.LEFT' in command:
        angle = np.deg2rad(-angle_change)  # Convert to radians
    else:
        return current_orientation

    # Calculate the new orientation using rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    new_orientation = np.dot(rotation_matrix, current_orientation)
    return new_orientation

def update_position(position, orientation, command):
    if 'Action.FORWARD' in command:
        return position + orientation
    elif 'Action.BACKWARD' in command:
        return position - orientation
    return position

# Load the command history file
command_history_path = 'command_history.npy'  # Replace with your file path
command_data = np.load(command_history_path, allow_pickle=True)

def build_path(command_history):
    positions = [np.array([0, 0])]  # Start at the origin
    orientation = np.array([0, -1])  # Initial orientation (Upwards)

    for key, value in command_history.item().items():
        command_str = str(value['command'])

        orientation = update_orientation(orientation, command_str)
        new_position = update_position(positions[-1], orientation, command_str)
        positions.append(new_position)

    return positions

def draw_path(positions, scale=5):
    max_coord = max(max(abs(pos[0]), abs(pos[1])) for pos in positions)
    img_size = int(max_coord * scale * 2) + 100
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # White background
    offset = np.array([img_size // 2, img_size // 2])  # Center the path

    for i in range(1, len(positions)):
        start = offset + positions[i - 1] * scale
        end = offset + positions[i] * scale
        cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), (0, 0, 0), 2)

    cv2.imshow('Path Map', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    positions = build_path(command_data)
    draw_path(positions)

if __name__ == "__main__":
    main()
