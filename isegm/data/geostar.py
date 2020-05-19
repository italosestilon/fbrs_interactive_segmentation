from pathlib import Path

import cv2
import numpy as np

from .base import ISDataset

class GeoStarDataset(ISDataset):
    def __init__(self, dataset_path, 
                images_dir_name="images", masks_dir_name="images-gt", markers_dir_name="images-labels",
                initial_markers=False,
                **kwargs):
        super(GeoStarDataset, self).__init__(**kwargs)

        self._initial_markers = initial_markers

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name
        if self._initial_markers:
            self._markers_path = self.dataset_path / markers_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}
        if self._initial_markers:
            self._markers_paths = {x.stem.split('-anno')[0]: x for x in self._markers_path.glob('*-anno.*')}

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])
        if self._initial_markers:
            markers_path = str(self._markers_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = np.max(cv2.imread(mask_path).astype(np.int32), axis=2)
        instances_mask[instances_mask  == 255] = 1
        instances_mask[instances_mask  == 128] = -1
        markers = None
        if self._initial_markers:
            markers = np.max(cv2.imread(markers_path).astype(np.int32), axis=2)
            markers[markers == 219] = 1
            markers[markers == 255] = 2

        instances_ids = [1]

        instances_info = {
            x: {'ignore': False}
            for x in instances_ids
        }

        return {
            'image': image,
            'instances_mask': instances_mask,
            'instances_info': instances_info,
            'initial_markers': markers,
            'image_id': index
        }
