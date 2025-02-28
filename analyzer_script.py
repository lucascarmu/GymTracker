import cv2
import os
from utils import analize_bbox, detect_squat_repetitions, create_video

def analize_squat_repetitions(input_video, output_path):
    tracker = cv2.legacy.TrackerBoosting_create()

    frame_numbers, bbox_heights, initial_frame = analize_bbox(video_path=input_video, output_path=output_path, tracker=tracker)

    start_frames, end_frames, peaks = detect_squat_repetitions(bbox_heights=bbox_heights,
                                                            frame_numbers=frame_numbers,
                                                            output_path=f"{output_path}{os.path.splitext(os.path.basename(input_video))[0]}")

    create_video(video_path=input_video, output_path=output_path,
                start_frames=start_frames, end_frames=end_frames,
                peaks=peaks, sec_pause=3, initial_frame=initial_frame)
    
if __name__ == "__main__":
    input_video = f"data/input/video_1.mp4" 
    output_path = f"data/output/"
    analize_squat_repetitions(input_video, output_path)