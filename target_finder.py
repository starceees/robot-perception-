import numpy as np
import cv2

# Initialize ORB detector
orb = cv2.ORB_create()

def detect_and_compute(image):
    """Detect and compute keypoints and descriptors in an image."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return orb.detectAndCompute(image, None)

def find_target_in_fpv(fpv_image, target_descriptor):
    """Check if a target image is found in the fpv image and return True if found."""
    kp1, des1 = detect_and_compute(fpv_image)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)

    # FLANN based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if des1 is not None and target_descriptor is not None:
        try:
            matches = flann.knnMatch(des1, target_descriptor, k=2)
            good_matches = []
            for match in matches:
                if len(match) == 2:  # Ensure that there are 2 matches
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            print(len(good_matches))
            if len(good_matches) > 50:  # Threshold for a good match
                return True
        except cv2.error as e:
            pass
    return False



def positions_are_close(pos1, pos2, threshold=5):
    """Check if two positions are within a certain threshold."""
    return np.linalg.norm(pos1 - pos2) < threshold

def update_orientation(current_orientation, command):
    angle_change = 2.5  # Angle change per command
    if 'Action.RIGHT' in command:
        angle = np.deg2rad(angle_change)
    elif 'Action.LEFT' in command:
        angle = np.deg2rad(-angle_change)
    else:
        return current_orientation

    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(rotation_matrix, current_orientation)

def update_position(position, orientation, command):
    if 'Action.FORWARD' in command:
        return position + orientation
    elif 'Action.BACKWARD' in command:
        return position - orientation
    return position

def build_path_and_plot_targets(command_history, target_images):
    positions = [np.array([0, 0])]
    orientation = np.array([0, -1])
    found_positions = [[] for _ in target_images]

    target_descriptors = [detect_and_compute(img)[1] for img in target_images]

    for key, value in command_history.items():
        command_str = str(value['command'])
        orientation = update_orientation(orientation, command_str)
        new_position = update_position(positions[-1], orientation, command_str)
        positions.append(new_position)

        if 'fpv' in value:
            fpv_image = value['fpv']
            for i, descriptor in enumerate(target_descriptors):
                if find_target_in_fpv(fpv_image, descriptor):
                    found_positions[i].append(new_position)

    # Check if any two positions are very close
    final_target_positions = []
    for pos_list in found_positions:
        for pos in pos_list:
            if any(positions_are_close(pos, other_pos) for other_pos in final_target_positions):
                continue
            final_target_positions.append(pos)

    return draw_path_with_targets(positions, final_target_positions)

def draw_path_with_targets(positions, target_positions, scale=2):
    max_coord = max(max(abs(pos[0]), abs(pos[1])) for pos in positions)
    img_size = int(max_coord * scale * 2) + 100
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    for i in range(1, len(positions)):
        start = (positions[i - 1] * scale + img_size // 2).astype(int)
        end = (positions[i] * scale + img_size // 2).astype(int)
        cv2.line(img, tuple(start), tuple(end), (0, 0, 0), 2)

    for target in target_positions:
        target_pixel = (target * scale + img_size // 2).astype(int)
        cv2.circle(img, tuple(target_pixel), 5, (0, 0, 255), -1)

    cv2.imshow('Path with Targets', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Load command history and target images
command_history_path = 'command_history.npy'
target_images_path = 'target_images.npy'

command_history = np.load(command_history_path, allow_pickle=True).item()
target_images = np.load(target_images_path, allow_pickle=True)

# Build path and plot targets
build_path_and_plot_targets(command_history, target_images)
