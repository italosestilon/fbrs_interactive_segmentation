from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker, Click

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

def _make_clicks_from_markers(markers):
    indices = np.where(markers != 0)
    labels = markers[indices]-1

    clicks = []

    for x_coord, y_coord, label in zip(indices[0], indices[1], labels):
        clicks.append(Click(label != 0, (x_coord, y_coord)))

    return clicks

def evaluate_dataset(dataset, predictor, oracle_eval=False, initial_markers=False, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        item = dataset[index]

        if oracle_eval:
            gt_mask = torch.tensor(sample['instances_mask'], dtype=torch.float32)
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
            predictor.opt_functor.mask_loss.set_gt_mask(gt_mask)

        initial_clicks = None
        if initial_markers:
            markers = sample['initial_markers']
            initial_clicks = _make_clicks_from_markers(markers)

        _, sample_ious, _ = evaluate_sample(item['images'], sample['instances_mask'], predictor, initial_clicks=initial_clicks, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(image_nd, instances_mask, predictor, max_iou_thr,
                    pred_thr=0.49, max_clicks=20, initial_clicks=None):
    clicker = Clicker(gt_mask=instances_mask, init_clicks=initial_clicks)
    pred_mask = np.zeros_like(instances_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image_nd)

        for click_number in range(max_clicks):
            if click_number > 0 or initial_clicks is None:
                clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            iou = utils.get_iou(instances_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
