from vis_nav_game import Player, Action
import pygame
import cv2
import matplotlib.pyplot as plt
import os
import threading
class KeyboardPlayerPyGame(Player):
    def extract_frames_from_video(self, video_path, output_dir, save_interval = 2):
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video file was opened successfully
        if not cap.isOpened():
            print("Error: Could not open video file")
            return

        frame_count = 0

        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # Break the loop if we have reached the end of the video
            if not ret:
                break

            if frame_count % save_interval == 0:
                # Save the frame as an image in the output directory
                frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
                cv2.imwrite(frame_filename, frame)

            frame_count += 1

        # Release the video file
        cap.release()

        print(f"Extracted {frame_count} frames and saved them in {output_dir}")
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        super(KeyboardPlayerPyGame, self).__init__()

        # Initialize video writer
        self.video_writer = None

    def init_video_writer(self, output_dir, output_filename, fps, frame_size):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, output_filename)

        # Define the codec and create a VideoWriter object with the .mp4 extension
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_dir, output_filename + ".mp4")

        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def save_frame(self, frame):
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def release_video_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()

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
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

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

        # match_img_left_right, match_img_front_back = create_map(targets[0], targets[1], targets[2], targets[3])

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        # plt.imshow(match_img_front_back)
        # plt.show()
        # plt.imshow(match_img_left_right)
        # plt.show()
        # Create a 2x2 grid
        # fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        # # Display each image in a grid cell
        # axes[0, 0].imshow(cv2.cvtColor(targets[0], cv2.COLOR_BGR2RGB))
        # axes[0, 0].set_title('Front Camera')
        #
        # axes[0, 1].imshow(cv2.cvtColor(targets[1], cv2.COLOR_BGR2RGB))
        # axes[0, 1].set_title('Back Camera')
        #
        # axes[1, 0].imshow(cv2.cvtColor(targets[2], cv2.COLOR_BGR2RGB))
        # axes[1, 0].set_title('Left Camera')
        #
        # axes[1, 1].imshow(cv2.cvtColor(targets[3], cv2.COLOR_BGR2RGB))
        # axes[1, 1].set_title('Right Camera')
        # plt.show()

        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        # Save the FPV image to a directory
        if self.fpv is not None:
            if self.video_writer is None:
                directory = "output_video"  # Change this to your desired directory path
                filename = "output_video.avi"  # Change this to your desired filename
                fps = 120  # Change the frame rate as needed
                frame_size = (w, h)
                self.init_video_writer(directory, filename, fps, frame_size)

            self.save_frame(self.fpv)

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

    def start_frame_extraction(self, video_path, output_dir):
        # Start a thread to extract frames from the video
        self.frame_extraction_thread = threading.Thread(target=self.extract_frames_from_video,
                                                        args=(video_path, output_dir))
        self.frame_extraction_thread.start()

if __name__ == "__main__":
    import vis_nav_game

    player = KeyboardPlayerPyGame()

    # Define the path to your .avi video file and the output directory
    video_path = r"C:\Users\raghu\course_env\vis_nav_player\output_video\output_video"  # Replace with the path to your .avi video file
    output_directory = "frames"  # Replace with the directory where you want to save the frames


    # Play the game
    vis_nav_game.play(the_player=player)

    # Release the video writer when the game ends
    player.release_video_writer()

    # Start frame extraction thread
    player.start_frame_extraction(video_path, output_directory)


    # Wait for the frame extraction thread to finish
    player.frame_extraction_thread.join()






