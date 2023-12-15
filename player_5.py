from vis_nav_game import Player, Action
import pygame
import cv2
import time
import numpy as np

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.prev_frame = None
        self.prev_descriptors = None
        self.consecutive_similar_frames = 0
        self.dead_end_threshold = 10
        self.command_history = {}
        self.command_index = 0
        self.is_exploration_phase = True
        self.last_command = None
        # Initialize ORB detector
        self.is_pre_navigation_done = False
        self.orb = cv2.ORB_create()
        self.is_moving = True
        self.is_turning = False
        self.navigation_continues = True
        self.max_degrees_left = 2  # Set maximum degrees for left turn
        self.max_degrees_right = 2  # Set maximum degrees for right turn
        self.max_backward_steps = 4
        self.is_deadend = False
        self.take_over_manual = False
        self.manual_control_enabled = False
        self.target_des = {}
        self.mode = 'exploration'
        self.navigation_iterator = None
        self.Event = None
        self.key_states = {}
        self.current_time = 0
        self.target_found_keys = []
        # FLANN parameters and object
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
            pygame.K_k: Action.BACKWARD
        }
    def toggle_mode(self):
        if self.mode == 'exploration':
            self.mode = 'manual'
        elif self.mode == 'manual':
            self.mode = 'exploration'
    def act(self):
        current_time = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                    if event.key == pygame.K_ESCAPE:
                        # Transition to navigation phase
                        self.is_exploration_phase = False
                        self.mode = 'navigation'
                        self.save_command_history()
                        print('Transitioning to Navigation Phase')
                        self.pre_navigation()
                        return Action.QUIT
                    elif event.key == pygame.K_k:
                        self.toggle_mode()
            self.Event = event

        # Call appropriate action method based on the current mode
        if self.mode == 'exploration':
            print("Exploration mode active")
            return  self.act_manual()
        elif self.mode == 'manual':
            print("Manual control active")
            return self.act_manual()
        elif self.mode == 'navigation':
            print("Navigation mode active")
            return self.act_navigate()


    def act_manual(self):
        # Reset action to IDLE at the beginning of each frame
        self.last_act = Action.IDLE

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            self.issue_command(Action.LEFT)
        if keys[pygame.K_RIGHT]:
            self.issue_command(Action.RIGHT)
        if keys[pygame.K_UP]:
            self.issue_command(Action.FORWARD)
        if keys[pygame.K_DOWN]:
            self.issue_command(Action.BACKWARD)

        self.record_command(self.last_act)

        return self.last_act
    def act_explore(self):
        self.current_time = time.time()
        # Frame processing for feature matching
        if self.fpv is not None:
            current_frame_gray = cv2.cvtColor(self.fpv, cv2.COLOR_BGR2GRAY)
            kp1, des1 = self.orb.detectAndCompute(current_frame_gray, None)
            print("we are entereing here")
            if self.prev_frame is not None and self.prev_descriptors is not None:
                print("Checking for dead end...")
                if des1 is not None and len(des1) > 0 and len(self.prev_descriptors) > 0:
                    try:
                        matches = self.flann.knnMatch(des1, self.prev_descriptors, k=2)
                    except cv2.error:
                        matches = []

                    # Filter for good matches
                    good_matches = []
                    for match in matches:
                        if len(match) == 2:  # Ensure that there are 2 matches
                            m, n = match
                            if m.distance < 0.9 * n.distance:
                                good_matches.append(m)
                    print(len(good_matches))

                    # Dead end detection based on number of good matches
                    if len(good_matches) <= 10:

                        if self.consecutive_similar_frames >= 200:  # Dead end detected
                            self.is_deadend = True
                            self.is_moving = False
                            #self.turn_right(10)
                        self.consecutive_similar_frames += 1
                    else:
                        # Here we can calculate F matrix, E matrix, decompose R and T, update pose, and call saving
                        self.consecutive_similar_frames = 0
                        self.is_deadend = True
                        self.is_moving = False

            self.prev_frame = current_frame_gray
            self.prev_descriptors = des1

        if self.is_moving:
            self.issue_command(Action.FORWARD)

        elif self.is_deadend:
            self.move_backward(self.max_backward_steps)
            self.turn_right(self.max_degrees_right)  # Assume you have a method to turn based on max degrees
            self.is_deadend = False

        else:
            self.turn_left(self.max_degrees_left)
            self.turn_right(self.max_degrees_right)

        # Record the command issued
        self.record_command(self.last_act)
        print(f"Consecutive Frames: {self.consecutive_similar_frames}")
        # Return the last action taken
        return self.last_act

    def move_backward(self, steps):
        print("Starting move_backward")
        while steps > 0:
            self.issue_command(Action.BACKWARD)
            # Assume you have a method here that waits for the robot to finish the backward movement by one step
            steps -= 1
        #self.issue_command(Action.IDLE)
        print("Finished move_backward")

    def turn_right(self, degrees):
        print(f"Turning right for {degrees} degrees")
        while degrees > 0:
            self.issue_command(Action.RIGHT)
            # Assume you have a method here that waits for the robot to finish the right turn by one degree
            degrees -= 1
        #self.issue_command(Action.IDLE)

    def turn_left(self, degrees):
        print(f"Turning left for {degrees} degrees")
        while degrees > 0:
            self.issue_command(Action.LEFT)
            # Assume you have a method here that waits for the robot to finish the left turn by one degree
            degrees -= 1
        #self.issue_command(Action.IDLE)


    def issue_command(self, command):
        # Update the last_act attribute with the current command
        self.last_act = command
        print(f"Issuing command: {command}")

    def record_command(self, command):
        # Record the command and its duration

        if self.last_command is not None:
            command_duration = self.current_time - self.command_start_time
            self.command_history[self.command_index] = {'command': self.last_command, 'duration': command_duration , 'fpv' : self.fpv.copy()}
            print(f"Command Duration : {command_duration}")
            self.command_index += 1
        self.last_command = command
        self.command_start_time = self.current_time

    def start_navigation(self):
        # Logic to be executed when switching to navigation phase
        self.pre_navigation()
    def issue_command(self, command):
        # Update the last_act attribute with the current command
        self.last_act = command
        print(f"Issuing command: {command}")
    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]

        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h / 2), 0), (int(h / 2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w / 2)), (h, int(w / 2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h / 2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w / 2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h / 2) + h_offset, int(w / 2) + w_offset), font, size, color, stroke,
                    line)
        command_history_path = 'command_history.npy'
        command_history = np.load(command_history_path, allow_pickle=True).item()
        self.build_path_and_plot_targets(command_history, targets)
        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)



    def save_command_history(self):
        np.save('command_history.npy', self.command_history)
        print('Command history saved.')
    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)

        self.show_target_images()

    def pre_exploration(self):
        K = self.get_camera_intrinsic_matrix()
        print(f'K={K}')

    ############# plot path
    def detect_and_compute(self, image):
        orb = cv2.ORB_create()
        """Detect and compute keypoints and descriptors in an image."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            test = orb.detectAndCompute(image, None)
        return test

    def find_target_in_fpv(self, fpv_image, target_descriptor):
        """Check if a target image is found in the fpv image and return True if found."""
        kp1, des1 = self.detect_and_compute(fpv_image)

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

    def positions_are_close(self, pos1, pos2, threshold=5):
        """Check if two positions are within a certain threshold."""
        return np.linalg.norm(pos1 - pos2) < threshold

    def update_orientation(self, current_orientation, command):
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

    def update_position(self, position, orientation, command):
        if 'Action.FORWARD' in command:
            return position + orientation
        elif 'Action.BACKWARD' in command:
            return position - orientation
        return position

    def build_path_and_plot_targets(self, command_history, target_images):
        positions = [np.array([0, 0])]
        orientation = np.array([0, -1])
        found_positions = [[] for _ in target_images]
        found_keys = []  # List to store the keys where targets are found

        target_descriptors = [self.detect_and_compute(img)[1] for img in target_images]

        for key, value in command_history.items():
            command_str = str(value['command'])
            orientation = self.update_orientation(orientation, command_str)
            new_position = self.update_position(positions[-1], orientation, command_str)
            positions.append(new_position)

            if 'fpv' in value:
                fpv_image = value['fpv']
                for i, descriptor in enumerate(target_descriptors):
                    if self.find_target_in_fpv(fpv_image, descriptor):
                        found_positions[i].append(new_position)
                        if key not in found_keys:  # Track the key if target is found
                            found_keys.append(key)

        # Consider only the lowest 2 keys
        self.target_found_keys = sorted(found_keys)[:2]
        print(f"Target Location at : {self.target_found_keys}")
        # Check if any two positions are very close
        final_target_positions = []
        for pos_list in found_positions:
            for pos in pos_list:
                if any(self.positions_are_close(pos, other_pos) for other_pos in final_target_positions):
                    continue
                final_target_positions.append(pos)

        return self.draw_path_with_targets(positions, final_target_positions)

    def draw_path_with_targets(self, positions, target_positions, scale=2):
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

        return cv2.imshow('Path with Targets', img)
    def pre_navigation(self):
        target_images = self.get_target_images()
        print("This is pre -navigation ")
        print(np.shape(target_images))
        # Load the command history for navigation
        if not self.is_exploration_phase:
            self.navigation_commands = np.load("command_history.npy", allow_pickle=True)
            #print(self.navigation_commands)
            self.navigation_iterator = iter(self.navigation_commands.item().items())
            print("Navigation Commands ")

    def act_navigate(self):
        if not self.navigation_continues:
            return Action.IDLE

        print(self.navigation_iterator)

        # Iterate over the command history
        try:
            key, command_info = next(self.navigation_iterator)

            # Check if the current key is in the target found keys
            if key in self.target_found_keys:
                print("Target reached. Stopping navigation commands.")
                self.navigation_continues = False  # Stop further navigation
                return Action.IDLE

            # If the key is not a target found key, execute the command
            self.issue_command(command_info['command'])
            return command_info['command']

        except StopIteration:
            # Return IDLE when all commands are executed
            return Action.IDLE
    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import logging

    logging.basicConfig(filename='vis_nav_player.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    import vis_nav_game as vng

    logging.info(f'player.py is using vis_nav_game {vng.core.__version__}')
    vng.play(the_player=KeyboardPlayerPyGame())