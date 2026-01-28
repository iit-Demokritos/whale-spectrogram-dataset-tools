from pathlib import Path
import json
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Callable
from PIL import Image, ImageOps
import copy
from src.utils import is_valid_file, parse_line_level_data, parse_page_level_data, aggregate_labels_info

class WhalesBaseDataset(Dataset, ABC):
    """
    Base Class for the Humpback Whale Song Spectrogram Dataset. It handles the core logic for loading and parsing the 
    data.

    Attributes:
        images_dir: Resolved path to the subset's image directory.
        labels_dir: Resolved path to the subset's label directory.
        image_paths: List of image file paths of teh subset.
        labels_data: Dictionary of all parsed labels. Image names serve as keys.      
    """
    def __init__(
        self, 
        dataset_dir: str | Path, 
        classes_filepath: str | Path | None = None,
        transform: Callable | None = None
    ):
        """        
        Args:
            dataset_dir: Root directory containing images (PNGs) and labels (JSONs).
            classes_filepath: Path to class definitions. It is optional, since it is required only for line-level datasets.
            transform: Transform(s) to be applied to the dataset.
        """
        super().__init__()
        self.dataset_dir = Path(dataset_dir)

        self.transform = transform

        # Get the paths to images and labels for the subset
        self.images_dir, self.labels_dir = self._get_paths()

        # Validate directories
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f'Image directory not found: {self.images_dir}\n'
            )
        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f'Labels directory not found: {self.labels_dir}\n'
            )
        
        # Load te images
        all_image_paths = [p for p in self.images_dir.rglob('*.png') if is_valid_file(p)]
        self.image_paths = sorted(all_image_paths, key=lambda path: path.name)

        # Load the labels
        labels = self._parse_all_labels()
        if labels is None:
            raise FileNotFoundError(f'No label files found in: {self.labels_dir}')
        self.labels_data = labels

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
            A tuple containing (images_dir, labels_dir).
        """
        pass

    @abstractmethod
    def _parse_labels(self, 
        labels_path: str | Path
    ) -> Dict[str, Any]:
        """
        Abstract method to parse a specific label file format.

        Args:
            labels_path: Path to the JSON file with the label(s).

        Returns:
            Dict: A dictionary where the key is the image filename and the value is its annotation data.
        """
        pass

    def _parse_all_labels(self):
        """Parses and aggregates all the labels info."""
        return aggregate_labels_info(self.labels_dir, self._parse_labels)
    
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

        if self.transform:
            image, labels = self.transform(image, labels)

        return image, labels   
    
class LineLevelDataset(WhalesBaseDataset):
    """Dataset loader for the line-level subset."""

    def __init__(
        self, 
        dataset_dir: str | Path, 
        classes_filepath: str | Path,
        transform: Callable | None = None
    ):  
        # Load the classes (.txt file) and construct the class map
        with open(classes_filepath) as f: 
            classes = [c.strip() for c in f if c.strip()]

        self.class_map = {cls: i for i, cls in enumerate(classes)}

        super().__init__(dataset_dir, classes_filepath, transform)

    def _get_paths(self) -> Tuple[Path, Path]:
        return self.dataset_dir / 'images' / 'lines', self.dataset_dir / 'labels' / 'line_level'
    
    def _parse_labels(self, 
        labels_path: str | Path
    ) -> Dict[str, Dict[str, List]]:
        """Parses line-level annotations, then maps the class names to integers to match model's (e.g. YOLO) requirements."""
        labels_info = parse_line_level_data(labels_path)

        for annotations in labels_info.values():
            annotations['unit_classes'] = [self.class_map[c] for c in annotations['unit_classes']]   
               
        return labels_info  

class PageLevelDataset(WhalesBaseDataset):
    """Dataset loader for the page-level subset."""
    def _get_paths(self):
        return self.dataset_dir / 'images' / 'pages', self.dataset_dir / 'labels' / 'page_level'

    def _parse_labels(self, 
        labels_path: str | Path
    ) -> Dict[str, List[float]]:
        return parse_page_level_data(labels_path)

