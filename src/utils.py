from pathlib import Path
import json
from typing import Dict, List, Callable

def is_valid_file(filepath: Path) -> bool:
    """Checks if the file is hidden, or if it is in a hidden directory (e.g. ".ipynb_checkpoints", ".git", etc.)."""
    check = lambda part: not part.startswith('.') or part == '..'
    return all(map(check, filepath.parts))

def parse_line_level_data( 
    labels_path: str | Path
) -> Dict[str, Dict[str, List]]:
    """
    Parses line-level JSON.

    Expected JSON structure:
    {   
        'general_image_name': filename.png
        'line_level_info': [
            {
                'image_name': 'filename-{line_id}.png',
                'line_id': int,
                'song_id': int,
                'unit_intervals': [[start, end], ...],
                'mute_intervals': [[start, end], ...],
                'unit_classes': [class_name, ...],
                'image_height': int,
                'image_width': int
            },
            ...
        ]
    }

    Args:
        labels_path: Path to the JSON.
    Returns:
        A dictionary where the key is the image filename and its values are the intervals and their respective unit 
        classes. It follows the structure:

        new_dict = {
            'filename.png': {
                'unit_intervals': [[start, end], ...],
                'unit_classes': [class_name, ...]
            }
        }
    """
    with open(labels_path) as js:
        data = json.load(js)

    labels_info = {}
    for entry in data['line_level_info']:
        image_name = entry['image_name']
        labels_info[image_name] = {
            'unit_intervals': entry['unit_intervals'],
            'unit_classes': entry['unit_classes']
        }

    return labels_info 

def parse_page_level_data(
        labels_path: str | Path
) -> Dict[str, List[float]]:
    """
    Parses page-level JSON data and extracts all line polygon coordinates.

    Expected JSON structure:
    {
        'image_name': 'filename.png',
        'image_height': int,
        'image_width': int,
        'polygons': [
            {   'line_id': int,
                'song_id': int,
                'points': [[x1, y1], [x2, y2], [x3, y3], ...]
            },
            ...
        ]
    }

    Args:
        labels_path: Path to the JSON.
    Returns:
        A dictionary where the key is the image name and its value is a list of all polygon coordinates. It follows 
        the structure:

        new_dict = {
            'filename.png': [
                [[x1, y1], [x2, y2], [x3, y3], ...],
                [[x1, y1], [x2, y2], [x3, y3], ...],
                ...
            ]
        }
    """
    with open(labels_path) as js:
        data = json.load(js)

    labels_info = {}
    image_name = data['image_name']
    poly_coords = []
    for pol in data['polygons']:
        poly_coords.append(pol['points'])
    
    labels_info[image_name] = poly_coords
    return labels_info

def aggregate_labels_info(
    labels_dir: Path,
    parser_func: Callable
) -> Dict[str, Dict[str, List]] | None:
    """
    Iterates over all JSON files in the labels directory, applies the parser_func to each, and aggregates the results
    into a general dictionary.

    Arguments:
        labels_dir: Path to the directory with the labels data, stored in JSON files.
        parser_func: Function to apply to the JSON filepaths ("parse_line_level_data", "parse_page_level_data", etc.)
    Returns:
        A dictionary with the aggregated labels info for all the JSON files.
    """
    label_paths = list(labels_dir.rglob('*.json'))
    if len(label_paths) == 0:
        print('No JSON files with labes found!')
        return None

    labels_info = {}
    for path in label_paths:
        if is_valid_file(path):
            # Update the dictionary with all the labels info (cache)
            labels_info |= parser_func(path)

    return labels_info