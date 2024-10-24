import cv2

def extract_first_frame(video_path, output_image_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if the video was successfully opened
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Get the number of frames in the video
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames in the video: {num_frames}")
    
    # Read the first frame
    success, frame = video_capture.read()
    
    if success:
        # Save the first frame as a JPG image
        cv2.imwrite(output_image_path, frame)
        print(f"First frame saved to {output_image_path}")
    else:
        print("Error: Could not read the first frame.")
    
    # Release the video capture object
    video_capture.release()

# Example usage
# video_path = "/local2/xingcheng/data/phyre/two_balls_within_template/train/00101:006/0-f.mp4"
# video_path = "/local2/xingcheng/data/phyre/ball_cross_template/test/00023:875/18-f.mp4"
video_path = "/local2/xingcheng/data/phyre/ball_cross_template/test/00007:008/7-f.mp4"
output_image_path = "/local2/xingcheng/Open-Sora/debug/condition/00007-008-7.jpg"
extract_first_frame(video_path, output_image_path)
