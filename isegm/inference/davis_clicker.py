from collections import namedtuple

from isegm.inference.davis_robot import InteractiveScribblesRobot
from isegm.inference import utils

import numpy as np

Click = namedtuple('Click', ['is_positive', 'coords'])

class DAVISClicker:
    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1):
        self.gt_mask = gt_mask
        self.ignore_label = ignore_label

        davis_robot = InteractiveScribblesRobot(min_nb_nodes=1)
        #scribbles = davis_robot.interact(pred, gt)
        #all_scribbles = scribbles
        #markers = utils.scribbles2mask(all_scribbles, image_shape[1:3]).squeeze()+1

        self.davis_robot = davis_robot

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)
    
    def add_click(self, click):
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)

    def _make_clicks_from_markers(self, markers):
        indices = np.where(markers != 0)
        labels = markers[indices]-1

        clicks = []

        for x_coord, y_coord, label in zip(indices[0], indices[1], labels):
            clicks.append(Click(label != 0, (x_coord, y_coord)))

        return clicks
    
    def _get_click(self, pred_mask):
        image_shape = self.gt_mask.shape
        scribbles = self.davis_robot.interact(pred_mask, self.gt_mask)
        markers = utils.scribbles2mask(scribbles, image_shape).squeeze()+1

        clicks = self._make_clicks_from_markers(markers)

        return clicks

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        clicks = self._get_click(pred_mask)

        for click in clicks:
            self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]
    
    def reset_clicks(self):
        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def __len__(self):
        return len(self.clicks_list)