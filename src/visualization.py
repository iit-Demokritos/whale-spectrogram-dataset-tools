import json
import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from src.utils import is_valid_file, parse_line_level_data, aggregate_labels_info

def convert_intervals_to_bboxes(
    intervals: List[List[float]], 
    image_height: int
) -> List[Tuple[int, int, int, int]]:
    """Converts time intervals [start, stop] to bounding boxes [x, y, w, h]."""
    y_min = int(0.05 * image_height)
    y_max = int(0.95 * image_height)
    height = y_max - y_min
    
    bboxes = []
    for x_min, x_max in intervals:
        width = x_max - x_min
        bboxes.append([x_min, y_min, width, height])

    return bboxes

def draw_annotations(
    image: np.ndarray, 
    labels: Dict[str, List],
) -> np.ndarray:
    """
    Draws bounding boxes and labels on a given image.
    
    Args:
        image: Numpy array (H, W, C).
        labels: Dictionary with 'unit_intervals' and 'unit_classes'.
        
    Returns:
        Annotated image (numpy array).
    """
    image = image.copy() 
    
    # Get bounding boxes
    intervals = labels['unit_intervals']
    image_height = image.shape[0]
    bboxes = convert_intervals_to_bboxes(intervals, image_height)

    # Get class names for units
    classes = labels['unit_classes']

    # Colors in BGR (openCV format)
    text_color = (0, 255, 255) # Yellow
    text_bg_color = (255, 0, 0) # Blue

    for (x, y, w, h), cls_name in zip(bboxes, classes):
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2) # Green box
        
        # Draw label
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(cls_name, font, font_scale, thickness)
        cv2.rectangle(image, (x, y + text_h + 10), (x + text_w, y), text_bg_color, -1) # background rectangle for label
        cv2.putText(image, cls_name, (x, y + text_h), font, font_scale, text_color, thickness, cv2.LINE_AA) # label
        
    return image

def save_multiple_line_level_annotations(
    images_dir: Path, 
    labels_dir: Path, 
    output_dir: Path
):
    """
    Loads multiple images and their labels, annotates them, and saves to output_dir.
    """
    # Load all the labels
    agg_labels_info = aggregate_labels_info(labels_dir, parse_line_level_data)
    if not agg_labels_info:
        print(f'No annotations found in "{labels_dir}" and its subdirectories! There is no reason to save anything to: \
              "{output_dir}"')
        return None
    
    # Load all the images
    image_paths = [p for p in images_dir.rglob('*.png') if is_valid_file(p)]
    if len(image_paths) == 0:
        print(f'No images found in "{images_dir}" and its subdirectories!')
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Processing and saving {len(image_paths)} images to "{output_dir}"')
    
    for i, img_path in enumerate(image_paths):
        if img_path.name not in agg_labels_info:
            continue
            
        # Load the image in BGR (OpenCV's format) so that we can draw colorful annotations
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Draw annotations
        labels = agg_labels_info[img_path.name]
        annotated_img = draw_annotations(img, labels) 
        
        # Save the produced image
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), annotated_img)
        
        if (i + 1) % 10 == 0:
            print(f"Saved {i + 1} images...")
    
    print('Done!')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Annotations Visualization')
    parser.add_argument(
        '--images_dir', 
        type=str, 
        required=True, 
        default='../data/whales_dataset/images/lines', 
        help='Path to prediction txt files'
    )
    parser.add_argument(
        '--labels_dir', 
        type=str, 
        required=True, 
        default='../data/whales_dataset/labels/line_level',
        help='Path to ground truth txt files'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='../data/visualized_annotations', 
        help='Output directory for the evaluation report file'
    )    
    return parser.parse_args()

def main():
    """Main function to run the script from the terminal."""
    args = parse_arguments()
    try:
        save_multiple_line_level_annotations(
            Path(args.images_dir),
            Path(args.labels_dir),
            Path(args.output_dir)
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")

if __name__ == "__main__":
    main()