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
        self.orb = cv2.ORB_create()
        self.is_moving = True
        self.is_turning = False
        self.backward_steps = 0
        self.right_steps = 0
        self.max_backward_steps = 100
        self.max_right_steps = 100
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
        self.prev_frame = None
        self.prev_descriptors = None
        self.consecutive_similar_frames = 0
        # self.command_history = {}
        # self.command_index = 0
        # self.is_exploration_phase = True

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        if self.is_exploration_phase:
            print("Now in Exploration!!!!!")
            return self.act_explore()
        else:
            print("Now in Navigation!!!!!")
            return self.act_navigate()


    def act_explore(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    if event.key == pygame.K_ESCAPE:
                        # Transition to the navigation phase
                        self.is_exploration_phase = False
                        # np.save(self.command_history_file, self.command_history)  # Save the command history
                        return Action.IDLE  # End exploration phase
                    self.last_act |= self.keymap[event.key]
                    current_command = self.keymap[event.key]
                else:
                    self.show_target_images()
        # Check the is_moving flag to determine movement direction
        if self.is_moving:
            print("Moving forward")
            self.issue_command(Action.FORWARD)
        else:
            print("Moving backward")
            self.issue_command(Action.BACKWARD)

        # For debugging: check the status of consecutive similar frames
        print(f"Consecutive similar frames: {self.consecutive_similar_frames}")

        # Process frames if we have started receiving them
        if self.fpv is not None:
            current_frame_gray = cv2.cvtColor(self.fpv, cv2.COLOR_BGR2GRAY)
            kp1, des1 = self.orb.detectAndCompute(current_frame_gray, None)

            if self.prev_frame is not None and self.prev_descriptors is not None:
                print("not going")
                if des1 is not None and len(des1) > 0 and len(self.prev_descriptors) > 0:
                    try:
                        matches = self.flann.knnMatch(des1, self.prev_descriptors, k=2)
                    except cv2.error:
                        matches = []

                    good_matches = []
                    for match in matches:
                        if len(match) == 2:  # Ensure that there are 2 matches
                            m, n = match
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)
                    print(len(good_matches) )
                    # Toggle the is_moving flag based on dead end detection
                    if len(good_matches) <= 1:
                        self.consecutive_similar_frames += 1
                        if self.consecutive_similar_frames >= 5:  # Dead end detected
                            self.is_moving = False
                            self.turn_right(100)
                    else:
                        self.consecutive_similar_frames = 0
                        self.is_moving = True

            self.prev_frame = current_frame_gray
            self.prev_descriptors = des1

        return self.last_act
    def record_command(self, command):
        current_time = time.time()
        if self.last_command is not None:
            command_duration = current_time - self.command_start_time
            self.command_history[self.command_index] = {'command': self.last_command, 'duration': command_duration}
            #print(self.command_history[self.command_index])
            self.command_index += 1

        self.last_command = command
        self.command_start_time = current_time
        # Save the command history at the end of the exploration phase
        if not self.is_exploration_phase:
            print('I think it is working ')
            # Convert the dictionary into a list of tuples for saving
            command_history_list = [(key, value['command'], value['duration']) for key, value in self.command_history.items()]
            np.save('command_history.npy', command_history_list)

    def move_backward(self, duration):
        print("Starting move_backward")
        self.is_moving = True
        start_time = time.time()
        while time.time() - start_time < duration:
            self.issue_command(Action.BACKWARD)
            print(f"Executing BACKWARD, time left: {duration - (time.time() - start_time)}")
        self.issue_command(Action.IDLE)
        self.is_moving = False
        print("Finished move_backward")

    def turn_right(self, degrees):
        for i in range(degrees):
            self.issue_command(Action.RIGHT)
            print(f"executed:{self.last_command}")
            #time.sleep(1)  # Adding a 1-second delay between each command
            #self.issue_command(Action.IDLE)  # Stop the turn
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

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def pre_exploration(self):
        K = self.get_camera_intrinsic_matrix()
        print(f'K={K}')

    def pre_navigation(self):
        # Load the command history for navigation
        if not self.is_exploration_phase:
            self.navigation_commands = np.load('command_history.npy', allow_pickle=True).item()
            self.navigation_index = 0

    def act_navigate(self):
        # Load the command history
        if not hasattr(self, 'navigation_commands'):
            # Load the command history from the .npy file
            command_history_list = np.load('command_history.npy', allow_pickle=True)

            # Check if the loaded array is not empty and its elements are tuples
            if command_history_list.size > 0 and isinstance(command_history_list[0], tuple):
                self.navigation_commands = [{'command': cmd, 'duration': dur} for _, cmd, dur in command_history_list]
            else:
                # Handle other formats or raise an error
                raise ValueError("Unexpected format in command history file.")

            self.navigation_index = 0

        # Check if there are more commands to execute
        if self.navigation_index < len(self.navigation_commands):
            command_info = self.navigation_commands[self.navigation_index]
            self.issue_command(command_info['command'])

            # Wait for the duration of the command
            time.sleep(command_info['duration'])

            # Increment the index for the next command
            self.navigation_index += 1
            return command_info['command']
        else:
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
