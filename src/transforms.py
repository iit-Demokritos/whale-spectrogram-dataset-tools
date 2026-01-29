from typing import Tuple, List, Dict, Any
from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms.functional as TF

class RandomSpectrogramLinePatcher:
    """
    Extracts a random patch of fixed-width from a spectrogram, finds the intersecting units, and adjust their 
    corresponding temporal annotations (time intervals).

    Attributes:
        patch_width: The target width of the patch in pixels.
        patch_height: The target height of the patch in pixels.
    """
    def __init__(
        self,
        patch_width: int=1024,
        patch_height: int=256,
    ):
        self.patch_width = patch_width
        self.patch_height = patch_height
        
    def __call__(
        self, 
        image: Image.Image,
        labels: Dict[str, List]
    ) -> Tuple[Image.Image, Dict[str, List]]:
        """
        Args:
            image: The input image (PIL).
            labels: Dictionary containing the annotations (keys: 'unit_intervals' and 'unit_classes').
        Returns:
            A tuple of:
                - Cropped or Padded Image (PIL).
                - Updated dictionary with adjusted 'unit_intervals' and 'unit_classes'.
        """
        img_width, img_height = image.size
        intervals = labels['unit_intervals']
        classes = labels['unit_classes']
        
        # In the following lines, we extract the random patch of the image, according to the following logic:
        # If the patch size is smaller that the image size (as expected in the most of the cases) to any direction, we 
        # crop the image along this direction. If, instead, it is greater, we pad the image with zeros to the right (for 
        # the width), or to the bottom (for the height). 
        patch_start_point = 0
        cropped_image = image.copy() # Initialize (to catch the case patch and image shape are equal), not cropped yet.
        
        # Width
        if self.patch_width < img_width:
        
            # Take a random start point for the crop
            patch_start_point = np.random.randint(0, img_width - self.patch_width)
            patch_stop_point = patch_start_point + self.patch_width
            cropped_image = cropped_image.crop((patch_start_point, 0, patch_stop_point, img_height)) # left, top, right, bottom
        
        elif self.patch_width > img_width:
            padding_size = self.patch_width - img_width
            cropped_image = ImageOps.expand(
                cropped_image,
                border=(0, 0, padding_size, 0), # left, top, right, bottom
                fill=(0, 0, 0)
            )

        # Height
        if self.patch_height < img_height:
            cropped_image = cropped_image.crop((0, 0, cropped_image.width, self.patch_height))

        elif self.patch_height > img_height:
            padding_size = self.patch_height - img_height
            cropped_image = ImageOps.expand(
                cropped_image,
                border=(0, 0, 0, padding_size),
                fill=(0, 0, 0)
            )
        
        # In the following lines, we identify which intervals are still inside the patch, and we adjust their new 
        # temporal extend (start and stop).
        new_intervals, new_classes = [], []
        for i, (start, stop) in enumerate(intervals):

            # Shift intervals
            new_start = start - patch_start_point
            new_stop = stop - patch_start_point

            if (new_stop > 0) and (new_start < self.patch_width):

                # Clamp the new intervals
                new_start = max(0, new_start)
                new_stop = min(new_stop, self.patch_width)

                new_intervals.append([new_start, new_stop])
                new_classes.append(classes[i])

        return cropped_image, {'unit_intervals': new_intervals, 'unit_classes': new_classes}
 
class ImageToTensor:
    """
    Transforms the PIL image of shape H x W x C and values in [0, 255] to a torch tensor of shape C x H x W and values 
    in [0, 1]. It does not handle "labels", they are just passed as input to maintain I/O consistency for the transforms.
    """
    def __call__(
        self, 
        image: Image.Image,
        labels: Dict[str, List]
    ) -> Tuple[torch.Tensor, Dict[str, List]]:
        """
        Args:
            image: The input image (PIL).
            labels: Dictionary containing the annotations (keys: 'unit_intervals' and 'unit_classes').
        Returns:
            A tuple of:
                - The tensorized input image.
                - The original labels unchanged.
        """
        image_tensor = TF.to_tensor(image)
        return image_tensor, labels

class UnitIntervalsToYOLO:
    """
    Converts temporal intervals (starts and stops) to normalized YOLO bounding boxes. Since the concern is one-dimension, 
    it creates bounding boxes that span the full height of the spectrogram (y=0.5, h=1.0).
    """
    def __call__(
        self, 
        image: torch.Tensor,
        labels: Dict[str, List]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            image: Input image tensor.
            labels Dictionary containing the annotations (keys: 'unit_intervals' and 'unit_classes').
        Returns:
            Tuple of the image (unchanged) and YOLO-formatted labels (keys changed to "bboxes" and "cls" to match the
            expected input data for Ultralytics YOLO models).
        """
        img_width = image.size()[-1] # tensor "image" is of shape CHW
        intervals = labels['unit_intervals']
        classes = labels['unit_classes']

        bboxes_in_norm_xywh = []
        for start, stop in intervals:
            x = (start + stop) / 2 / img_width
            y = .5
            w = (stop - start) / img_width
            h = 1
            bboxes_in_norm_xywh.append([x, y, w, h])
    
        if len(bboxes_in_norm_xywh) == 0:
            classes_tensor = torch.empty(0, dtype=torch.int64)
            bboxes_tensor = torch.empty(0, 4, dtype=torch.float32)
        else:
            classes_tensor = torch.tensor(classes, dtype=torch.int64)
            bboxes_tensor = torch.tensor(bboxes_in_norm_xywh, dtype=torch.float32)

        return image, {'bboxes': bboxes_tensor, 'cls': classes_tensor} 
    
class ComposeTransforms:
    """Composes a sequence of transforms that accept (image, labels) pairs."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(
        self, 
        image: Image.Image | torch.Tensor, 
        labels: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        for t in self.transforms:
            image, labels = t(image, labels)
        return image, labels

