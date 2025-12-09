import torch
from typing import Tuple, List, Dict
from PIL import Image, ImageOps
import numpy as np

class RandomSpectrogramLinePatcher:
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
        
        img_width, img_height = image.size
        intervals = labels['unit_intervals']
        classes = labels['unit_classes']
       
        # In the following lines, we extract the random patch of the image, according to the following logic:
        # If the patch size is smaller that the image size (as expected in the most of the cases) to any direction, we 
        # crop the image along this direction. If, instead, it is greater, we pad the image with zeros to the right (for 
        # the width), or to the bottom (for the height). 
        patch_start_point = 0
        if self.patch_width < img_width:
        
            # Take a random x-coordinate for the crop
            patch_start_point = np.random.randint(0, img_width - self.patch_width)
            patch_stop_point = patch_start_point + self.patch_width
            image = image.crop((patch_start_point, 0, patch_stop_point, img_height)) # left, top, right, bottom
        
        elif self.patch_width > img_width:
            padding_size = self.patch_width - img_width
            image = ImageOps.expand(
                image,
                border=(0, 0, padding_size, 0), # left, top, right, bottom
                fill=(0, 0, 0)
            )

        if self.patch_height < img_height:
            image = image.crop((0, 0, image.width, self.patch_height))

        elif self.patch_height > img_height:
            padding_size = self.patch_height - img_height
            image = ImageOps.expand(
                image,
                border=(0, 0, 0, padding_size),
                fill=(0, 0, 0)
            )
        
        # In the following lines, we identify which intervals are still inside the patch, and we adjust their new 
        # coordinates.
        new_intervals, new_classes = [], []
        for i, (start, stop) in enumerate(intervals):

            # Shift x-coordinates
            new_start = start - patch_start_point
            new_stop = stop - patch_start_point

            if (new_stop > 0) and (new_start < self.patch_width):

                # Clamp the new coordinates
                new_start = max(0, new_start)
                new_stop = min(new_stop, self.patch_width - 1)

                new_intervals.append([new_start, new_stop])
                new_classes.append([classes[i]])

        return image, {'unit_intervals': new_intervals, 'unit_classes': new_classes}


        

class UnitIntervalsToYOLO:
    def __init__(
        self,
        labels: Dict[str, Dict[str, List]]
    ):
        pass


