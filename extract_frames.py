import cv2
import os

def extract_frames_from_video(video_path, output_dir):
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

if __name__ == "__main__":
    video_path = r"C:\Users\raghu\course_env\vis_nav_player\output_video\output_video.avi.mp4" # Replace with the path to your .avi video file
    output_directory = "frames"  # Replace with the directory where you want to save the frames

    extract_frames_from_video(video_path, output_directory)
