import cv2
import os

class FrameExtractor:
    def __init__(self, video_path, output_folder, frame_interval=20):
        """
        Extract frames from a video at a specified interval.

        Args:
            video_path (str): Path to the video file.
            output_folder (str): Directory where extracted frames will be saved.
            frame_interval (int): Number of frames to skip between extractions (default: 20).
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.frame_interval = frame_interval

        os.makedirs(self.output_folder, exist_ok=True)

    def extract_frames(self):
        """Extract frames from the video at the given interval and save them as images."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {self.video_path}")
            return

        frame_count = 0
        saved_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % self.frame_interval == 0:
                frame_path = os.path.join(self.output_folder, f"frame_{saved_count:04d}.png")
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Extraction complete! {saved_count} frames saved in '{self.output_folder}'.")


if __name__ == "__main__":
    video_path = "skippy.mp4"
    output_folder = "extracted_frames"
    extractor = FrameExtractor(video_path, output_folder, frame_interval=1)
    extractor.extract_frames()
