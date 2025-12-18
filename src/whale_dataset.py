from pathlib import Path
import json
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Callable
from PIL import Image, ImageOps
import copy

def _is_valid_file(filepath: Path) -> bool:
    """Helper function to filter any hidden .txt files, or files that are in hidden folders (e.g. ".ipynb_checkpoints", 
    ".git", etc.)"""

    check = lambda part: not part.startswith('.') or part == '..'
    return all(map(check, filepath.parts))


class WhalesBaseDataset(Dataset, ABC):
    """
    Base Class for the Humpback Whale Song Spectrogram Dataset. It handles the core logic for loading and parsing the 
    data.

    Attributes:
        image_dir: Resolved path to the subset's image directory.
        label_dir: Resolved path to the subset's label directory.
        image_paths: List of image file paths of teh subset.
        labels_data: Dictionary of all parsed labels. Image names serve as keys.      
    """
    def __init__(
        self, 
        dataset_dir: str | Path, 
        classes_filepath: str | Path,
        transform: Callable | None = None
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)

        with open(classes_filepath) as f: # NOTE: .txt file
            classes = [c.strip() for c in f if c.strip()]

        self.class_map = {cls: i for i, cls in enumerate(classes)}
        self.transform = transform

        # Get the paths to images and labels for the subset
        self.image_dir, self.label_dir = self._get_paths()

        # Validate directories
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f'Image directory not found: {self.image_dir}\n'
            )
        if not self.label_dir.exists():
            raise FileNotFoundError(
                f'Labels directory not found: {self.label_dir}\n'
            )
        
        # Load te images
        all_image_paths = [p for p in self.image_dir.rglob('*.png') if _is_valid_file(p)]
        self.image_paths = sorted(all_image_paths, key=lambda path: path.name)

        # Load the labels
        self.labels_data = self._parse_all_labels()

        # Validate 1:1 correspondence between images and label
        for img_path in self.image_paths:
            if img_path.name not in self.labels_data:
                raise FileNotFoundError(
                    f'No labels found for image: {img_path}\n'
                )
           
    @abstractmethod
    def _get_paths(self) -> Tuple[Path, Path]:
        """
        Abstract method to get the specific image and label directories.

        Returns:
            A tuple containing (image_dir, label_dir).
        """
        pass

    @abstractmethod
    def _parse_label(self, 
        label_path: str | Path
    ) -> Dict[str, Any]:
        """
        Abstract method to parse a specific label file format.

        Args:
            label_path: Path to the JSON file with the label(s).

        Returns:
            Dict: A dictionary where the key is the image filename and the value is its annotation data.
        """
        pass

    def _parse_all_labels(self):
        """Iterates over all JSON files in the label directory and aggregates them into a single dictionary."""
        label_paths = self.label_dir.rglob('*.json')
        labels_data = {}
        for path in label_paths:
            if _is_valid_file(path):

                # Update the dictionary with all the label info (cache)
                labels_data |= self._parse_label(path)

        return labels_data
    
    def __len__(self) -> int:
        """Returns the total number of samples of the dataset."""
        return len(self.image_paths)
        
    def __getitem__(self, index: int) -> Tuple[Image.Image, Any]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            A Tuple of (image and label). Applies a transform, if one is specified.
        """
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB') # Convert grayscale -> RGB, to fit with the most ML frameworks.
        image = ImageOps.invert(image) # Negate the image (white foreground/zero background). Still in [0, 255] (uint8).
        labels = copy.deepcopy(self.labels_data[image_path.name])

        labels['unit_classes'] = [
            self.class_map[c] for c in labels['unit_classes']
        ]

        if self.transform:
            image, labels = self.transform(image, labels)

        return image, labels   
    
class LineLevelDataset(WhalesBaseDataset):
    """Dataset loader for the line-level subset."""
    def _get_paths(self) -> Tuple[Path, Path]:
        return self.dataset_dir / 'images' / 'lines', self.dataset_dir / 'labels' / 'line_level'
    
    def _parse_label(self, 
        label_path: str | Path
    ) -> Dict[str, Dict[str, List]]:
        """
        Parses line-level JSON data and reformats it for easy lookup by line image name.

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
            label_path: Path to the JSON.
        Returns:
            A dictionary where the key is the image file name and its values are the intervals and their respective unit 
            classes. It follows the structure:

            new_dict = {
                'filename-{line_id}.png': {
                    'unit_intervals': [[start, end], ...],
                    'unit_classes': [class_name, ...]
                }
            }
        """
        with open(label_path) as js:
            data = json.load(js)

        labels_info = {}
        for entry in data['line_level_info']:
            image_name = entry['image_name']
            labels_info[image_name] = {
                'unit_intervals': entry['unit_intervals'],
                'unit_classes': entry['unit_classes']
            }

        return labels_info 

class PageLevelDataset(WhalesBaseDataset):
    """Dataset loader for the page-level subset."""
    def _get_paths(self):
        return self.dataset_dir / 'images' / 'pages', self.dataset_dir / 'labels' / 'page_level'

    def _parse_label(self, 
        label_path: str | Path
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
            label_path: Path to the JSON.
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
        with open(label_path) as js:
            data = json.load(js)

        labels_info = {}
        image_name = data['image_name']
        poly_coords = []
        for pol in data['polygons']:
            poly_coords.append(pol['points'])
        
        labels_info[image_name] = poly_coords
        return labels_info

