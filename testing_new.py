import numpy as np

# Replace this with the path to your .npy file
file_path = r"C:\Users\raghu\course_env\vis_nav_player\command_history.npy"

# Load the file
command_history = np.load(file_path, allow_pickle=True)

print(command_history)
fpv_images = []
for key, value in command_history.item().items():
    command = value['command']
    duration = value['duration']
    fpv = value['fpv']
    fpv_images.append(fpv)
    print(f"Key: {key}, Command: {command}, Duration: {duration}")
