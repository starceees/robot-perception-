from vis_nav_game import Player, Action
import pygame
import cv2
import os
import threading

class KeyboardPlayerPyGame(Player):
    def extract_frames_from_video(self, video_path, output_dir):
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
        self.captured_frames = []  # List to store captured frames
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

    def dfs(self, frame, visited, map):
        # Mark the current frame as visited
        visited[frame] = True

        # Analyze the frame to gather information about the environment
        # Update the map based on the analysis

        # Explore neighboring frames
        for neighbor_frame in self.get_neighbors(frame):
            if not visited[neighbor_frame]:
                self.dfs(neighbor_frame, visited, map)

    def build_map_from_frames(self):
        num_frames = len(self.captured_frames)
        visited = [False] * num_frames

        # Choose a starting frame (you may modify this logic)
        start_frame = 0

        # Initialize the map (you may modify this logic)
        map = self.initialize_map(num_frames)

        # Start DFS from the initial frame
        self.dfs(start_frame, visited, map)

        # You can now use the 'map' to represent the environment based on the captured frames.
        return map

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        # Save the FPV image to a directory and capture the frame
        self.save_frame(fpv)
        self.captured_frames.append(fpv)  # Add the frame to the captured frames list

        # Build the map from the captured frames (you can call this method as needed)
        map = self.build_map_from_frames()

        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1]
            shape = opencv_image.shape[1::-1]
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

    def get_neighbors(self, frame):
        # Define how to obtain neighboring frames from the current frame
        # You can implement this logic based on your video setup
        # For simplicity, you can return a list of frame indices representing neighbors
        pass

    def initialize_map(self, num_frames):
        # Initialize the map data structure
        # You can adapt this based on the specifics of your map representation
        pass

if __name__ == "__main__":
    import vis_nav_game

    player = KeyboardPlayerPyGame()

    # Define the path to your .avi video file and the output directory
    video_path = r"C:\Users\raghu\course_env\vis_nav_player\output_video\output_video.avi"  # Replace with the path to your .avi video file
    output_directory = "frames"  # Replace with the directory where you want to save the frames

    # Play the game
    vis_nav_game.play(the_player=player)

    # Release the video writer when the game ends
    player.release_video_writer()

    # Start frame extraction thread
    player.start_frame_extraction(video_path, output_directory)

    # Wait for the frame extraction thread to finish
    player.frame_extraction_thread.join()
