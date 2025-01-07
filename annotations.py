# this file is the preprocessing of the images, including annotating and saving the images and annotations into pairs

import cv2
import os
import shutil
import re
from PIL import Image

video_path = "./200_06/VIRAT_S_000200_06_001693_001824.mp4"
output_folder = "./200_06/frames"
pairs_dir = "./200_06/annotation_pairs"
interval = 60


# cut videos into frames
def extract_frames(video_path, output_folder):
    print(f"Start processing video: {video_path}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist: {video_path}")
        return
        
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Read video
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return
    
    # Get video information
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video information:")
    print(f"- Total frames: {total_frames}")
    print(f"- FPS: {fps}")
    print(f"- Resolution: {width}x{height}")
    
    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
            
        frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count:04d}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)
        
        # Save frame and verify
        save_success = cv2.imwrite(frame_path, frame)
        if not save_success:
            print(f"Warning: Cannot save frame {frame_count} to {frame_path}")
        
        frame_count += 1
        if frame_count % 100 == 0:  # Print progress every 100 frames
            print(f"Processed {frame_count} frames")
    
    video.release()
    print(f"Completed! Extracted {frame_count} frames")


# draw bounding box on each frames
class BoundingBoxDrawer:
    def __init__(self, image_folder, output_folder="annotations", sample_interval=60):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.image = None
        self.original_image = None
        self.current_box = None
        self.sample_interval = sample_interval
        self.current_filename = ""
        self.current_frame_number = 0

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created annotation output folder: {output_folder}")

    def get_frame_number(self, filename):
        # Extract frame number from filename (assuming format contains "frame_XXXX")
        try:
            return int(filename.split('frame_')[1].split('.')[0])
        except:
            return 0

    def save_box_to_file(self, class_number):
        if self.current_box is None:
            return
            
        # Get current image filename (without extension)
        base_name = os.path.splitext(self.current_filename)[0]
        output_path = os.path.join(self.output_folder, f"{base_name}.txt")
        
        # Save box coordinates to file (append mode)
        x, y, w, h = self.current_box
        frame_number = self.get_frame_number(self.current_filename)
        with open(output_path, 'a') as f:
            # Format: 0 0 frame_number x y width height
            f.write(f"0 0 {frame_number} {x} {y} {w} {h} {class_number}\n")
        print(f"Added annotation data to: {output_path}")

    def draw_box(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.original_image.copy()
                cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.putText(img_copy, self.current_filename, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.image = img_copy

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            width = abs(x - self.ix)
            height = abs(y - self.iy)
            top_left_x = min(self.ix, x)
            top_left_y = min(self.iy, y)
            
            self.current_box = (top_left_x, top_left_y, width, height)
            print(f"Frame: {self.current_filename}")
            print(f"Box position: x={top_left_x}, y={top_left_y}, width={width}, height={height}")

            class_number = None

            while True:
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    class_number = int(chr(key))
                    print(f"Class number selected: {class_number}")
                    break
                elif key == ord('q'):
                    print("Exiting without saving box.")
                    self.current_box = None
                    return
            
            # Save annotation data
            self.save_box_to_file(class_number)
            
            # Draw the box on the original image
            cv2.rectangle(self.original_image, (top_left_x, top_left_y), 
                        (top_left_x + width, top_left_y + height), (0, 255, 0), 2)
            self.image = self.original_image.copy()

    def process_images(self):
        image_files = [f for f in os.listdir(self.image_folder) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        image_files.sort()

        # Sample every Nth frame
        image_files = image_files[::self.sample_interval]
        print(f"Total sampled images: {len(image_files)}")

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.draw_box)

        current_idx = 0
        while current_idx < len(image_files):
            self.current_filename = image_files[current_idx]
            image_path = os.path.join(self.image_folder, self.current_filename)
            self.original_image = cv2.imread(image_path)
            self.image = self.original_image.copy()
            
            # Check for existing annotation file and load boxes
            base_name = os.path.splitext(self.current_filename)[0]
            annotation_file = os.path.join(self.output_folder, f"{base_name}.txt")
            if os.path.exists(annotation_file):
                status = "Annotated"
                # Show existing boxes
                with open(annotation_file, 'r') as f:
                    for line in f:
                        # Parse line: 0 0 frame x y w h
                        parts = line.strip().split()
                        if len(parts) == 7:  # Ensure correct format
                            x, y, w, h = map(int, parts[3:])
                            cv2.rectangle(self.original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        elif len(parts) == 8:  # Ensure correct format
                            x, y, w, h, c = map(int, parts[3:])
                            cv2.rectangle(self.original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.image = self.original_image.copy()
            else:
                status = "Not annotated"
            
            # Show filename and annotation status
            cv2.putText(self.image, f"{self.current_filename} - {status}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            while True:
                cv2.imshow('Image', self.image)
                key = cv2.waitKey(1) & 0xFF

                if key == 83 or key == ord('d'):  # Right arrow or 'n'
                    current_idx += 1
                    break
                elif key == 81 or key == ord('a'):  # Left arrow or 'p'
                    current_idx = max(0, current_idx - 1)
                    break
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord('c'):
                    self.image = self.original_image = cv2.imread(image_path)
                    cv2.putText(self.image, f"{self.current_filename} - {status}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.current_box = None
                    # Delete annotation file
                    if os.path.exists(annotation_file):
                        os.remove(annotation_file)
                        print(f"Deleted annotation file: {annotation_file}")

        cv2.destroyAllWindows()


# standardize the pairs
def standardize_pairs(annotations_dir, pairs_dir):
    os.makedirs(pairs_dir, exist_ok=True)

    # Dictionary to hold annotations grouped by prefix
    # key: prefix (e.g., "S_000201_01_")
    # value: list of tuples (frame_num, filename)
    groups = {}

    # Regex to parse the frame number and parts of the filename
    pattern = re.compile(r"^VIRAT_(S_\d+_\d+)_.+_frame_(\d{4})\.txt$")

    for filename in os.listdir(annotations_dir):
        if filename.endswith(".jpg"):
            # Full path of the annotation file
            filepath = os.path.join(annotations_dir, filename)
            
            # Example filename format:
            # VIRAT_S_000201_01_000384_000589_frame_0000.txt
            # We want to extract:
            # prefix: S_000201_01_
            # frame: 0000 (as int: 0)

            # Split by underscore to identify parts
            parts = filename.split('_')
            # parts example:
            # [ 'VIRAT', 'S', '000201', '01', '000384', '000589', 'frame', '0000.txt' ]

            # Extract frame number
            # The frame number is in parts[7].strip('.txt')
            frame_str = parts[7].split('.')[0]  # "0000"
            frame_num = int(frame_str)

            if (frame_num % 60) != 0:
                continue

            # Extract prefix: S_000201_01_
            # According to the example, we only keep up to "S_000201_01_"
            # That would be parts[1], parts[2], parts[3]
            prefix = f"{parts[1]}_{parts[2]}_{parts[3]}_"
            
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((frame_num, filepath))

    # Now for each prefix, we sort by frame number and pair them
    for prefix, entries in groups.items():
        # Sort by frame number
        entries.sort(key=lambda x: x[0])

        # entries is now like [(0, 'path_to_frame_0000.txt'), (60, 'path_to_frame_0060.txt'), (120, '...'), ...]

        # We'll pair them in increments of 60 frames
        # That is, if we have frames [0, 60, 120, 180 ...],
        # we create pairs (0,60), (60,120), (120,180), etc.

        # Since we know each pair differs by 60 frames, we can step through:
        for i in range(len(entries) - 1):
            frame_num_1, file_1 = entries[i]
            frame_num_2, file_2 = entries[i+1]

            # Check if the second frame is exactly 60 frames after the first
            if frame_num_2 - frame_num_1 == 60:
                # Format the frame numbers with zero padding to 4 digits
                frame_str_1 = f"{frame_num_1:04d}"
                frame_str_2 = f"{frame_num_2:04d}"

                # Construct the pair directory name
                pair_dir_name = f"Pair_{prefix}{frame_str_1}_{frame_str_2}"
                pair_dir_path = os.path.join(pairs_dir, pair_dir_name)

                # Create the pair directory if not exists
                os.makedirs(pair_dir_path, exist_ok=True)

                # Copy the two annotation files into this directory
                shutil.copy(file_1, pair_dir_path)
                shutil.copy(file_2, pair_dir_path)

    # REnaming
    # Iterate through all subdirectories in the pairs directory
    for pair_dir_name in os.listdir(pairs_dir):
        pair_dir_path = os.path.join(pairs_dir, pair_dir_name)

        # Ensure it's a directory
        if os.path.isdir(pair_dir_path):
            # Iterate through all files in the pair directory
            for filename in os.listdir(pair_dir_path):
                # Check if the filename starts with "VIRAT"
                if filename.startswith("VIRAT"):
                    # Construct the new filename by removing "VIRAT_"
                    new_filename = filename.replace("VIRAT_", "", 1)

                    # Full paths to the old and new file names
                    old_file_path = os.path.join(pair_dir_path, filename)
                    new_file_path = os.path.join(pair_dir_path, new_filename)

                    # Rename the file
                    os.rename(old_file_path, new_file_path)

    print("Renaming of files starting with 'VIRAT' completed.")

    # convert to PNG
    # Iterate through all subdirectories in the pairs directory
    for pair_dir_name in os.listdir(pairs_dir):
        pair_dir_path = os.path.join(pairs_dir, pair_dir_name)

        # Ensure it's a directory
        if os.path.isdir(pair_dir_path):
            # Iterate through all files in the pair directory
            for filename in os.listdir(pair_dir_path):
                # Check if the filename starts with "VIRAT" and rename
                if filename.startswith("VIRAT"):
                    # Construct the new filename by removing "VIRAT_"
                    new_filename = filename.replace("VIRAT_", "", 1)
                    old_file_path = os.path.join(pair_dir_path, filename)
                    new_file_path = os.path.join(pair_dir_path, new_filename)
                    os.rename(old_file_path, new_file_path)
                    filename = new_filename

                # Convert .jpg files to .png
                if filename.endswith(".jpg"):
                    jpg_path = os.path.join(pair_dir_path, filename)
                    png_path = os.path.join(pair_dir_path, filename.replace(".jpg", ".png"))

                    # Open the JPG image and save it as PNG
                    with Image.open(jpg_path) as img:
                        img.save(png_path, "PNG")

                    # Remove the original JPG file
                    os.remove(jpg_path)

    print("Renaming and JPG to PNG conversion completed.")


extract_frames(video_path, output_folder)
drawer = BoundingBoxDrawer(output_folder, output_folder="./200_06/annotations", sample_interval=interval)
drawer.process_images()
standardize_pairs("./200_06/annotations", pairs_dir)