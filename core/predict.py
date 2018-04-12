import os
import sys
import glob
import random
from mask_rcnn import model as modellib
from core.mask_rcnn_config import MyMaskRcnnConfig, TEST_DATA_DIR
from core.utils import georeference, rectangularize, get_contours, get_contour
from typing import Iterable, Tuple, List
from PIL import Image
from core.settings import IMAGE_WIDTH
import numpy as np
import json
import cv2
import math


class Predictor:
    config = MyMaskRcnnConfig()

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 30
        IMAGE_MIN_DIM = 320
        IMAGE_MAX_DIM = 320

    def __init__(self, weights_path: str):
        if not os.path.isfile(weights_path):
            raise RuntimeError("Weights cannot be found at: {}".format(weights_path))
        self.weights_path = weights_path
        self._model = None

    def predict_arrays(self, images: List[Tuple[np.ndarray, str]], extent=None, do_rectangularization=True, tile=None, verbose=1) \
            -> List[List[Tuple[int, int]]]:
        if not tile:
            tile = (0, 0)

        BATCH_SIZE = 30

        if not self._model:
            print("Loading model")
            inference_config = self.InferenceConfig()
            print("Predicting {} images".format(len(images)))
            # Create model in training mode
            model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="log")
            model.load_weights(self.weights_path, by_name=True)
            self._model = model

        all_prediction_results = []
        model = self._model
        batches = math.ceil(len(images) / BATCH_SIZE)
        for i in range(batches):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            img_with_id_batch = images[start:end]
            print("Predicting batch {}/{}".format(i, batches))
            img_batch = list(map(lambda i: i[0], img_with_id_batch))
            id_batch = list(map(lambda i: i[1], img_with_id_batch))
            results = model.detect(img_batch, verbose=verbose)
            # result_with_image_id = []
            for i, res in enumerate(results):
                all_prediction_results.append((res, id_batch[i]))
            # all_prediction_results.extend(results)
        print("Extracting contours...")
        point_sets = []
        for res, coco_img_id in all_prediction_results:
            masks = res['masks']
            for i in range(masks.shape[-1]):
                mask = masks[:, :, i]
                points = get_contour(mask)
                score = res['scores'][i]
                point_sets.append((list(points), score, coco_img_id))
        print("Contours extracted")
        return point_sets

    def predict_path(self, img_path: str, extent=None, verbose=1) -> List[List[Tuple[int, int]]]:
        return self.predict_paths([img_path], extent=extent, verbose=verbose)

    def predict_paths(self, all_paths: List[str], extent=None, verbose=1) -> List[List[Tuple[int, int]]]:
        all_images = []
        for p in all_paths:
            #img = Image.open(p)
            #data = np.asarray(img, dtype="uint8")
            data = cv2.imread(p)
            coco_img_id = int(os.path.basename(p).replace(".jpg", ""))
            all_images.append((data, coco_img_id))
        return self.predict_arrays(images=all_images, extent=extent, verbose=verbose)


def test_images(annotations_file_name="predictions.json", processed_images_name="tested_images.txt", nr_images=None, target_dir=TEST_DATA_DIR):
    predictor = Predictor(os.path.join(os.getcwd(), "model", "osm20180407T1301", "mask_rcnn_osm_0232.h5"))
    annotations_path = os.path.join(os.getcwd(), annotations_file_name)
    images = glob.glob(os.path.join(target_dir, "**/*.jpg"), recursive=True)
    if nr_images:
        # random.shuffle(images)
        images = images[:nr_images]
    annotations = []
    if os.path.isfile(annotations_path):
        with open(annotations_path, 'r', encoding="utf-8") as f:
            data = f.read()
            if data:
                annotations = json.loads(data)

    point_sets_with_score = predictor.predict_paths(images, verbose=0)

    for contour, score, coco_img_id in point_sets_with_score:
        xs = list(map(lambda pt: int(pt[0])-0, contour))  # -10 padding
        ys = list(map(lambda pt: int(pt[1])+0, contour))
        if contour:
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        else:
            bbox = []
        points_sequence = []
        for idx, x in enumerate(xs):
            points_sequence.append(x)
            points_sequence.append(ys[idx])
        ann = {
            "image_id": coco_img_id,
            "category_id": 100,
            "segmentation": [points_sequence],
            "bbox": bbox,
            "score": float(np.round(score, 2))
        }
        if bbox:
            annotations.append(ann)
            with open(annotations_path, "w") as fp:
                fp.write(json.dumps(annotations))


if __name__ == "__main__":
    test_images(nr_images=4)
    # test_images()
