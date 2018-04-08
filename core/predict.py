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


class Predictor:
    config = MyMaskRcnnConfig()

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_MIN_DIM = 320
        IMAGE_MAX_DIM = 320

    def __init__(self, weights_path: str):
        if not os.path.isfile(weights_path):
            raise RuntimeError("Weights cannot be found at: {}".format(weights_path))
        self.weights_path = weights_path
        self._model = None

    def predict_array(self, img_data: np.ndarray, extent=None, do_rectangularization=True, tile=None, verbose=1) \
            -> List[List[Tuple[int, int]]]:
        if not tile:
            tile = (0, 0)

        if not self._model:
            print("Loading model")
            inference_config = self.InferenceConfig()
            # Create model in training mode
            model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="log")
            model.load_weights(self.weights_path, by_name=True)
            self._model = model

        model = self._model
        print("Predicting...")
        res = model.detect([img_data], verbose=verbose)
        print("Prediction done")
        print("Extracting contours...")
        point_sets = []
        masks = res[0]['masks']
        for i in range(masks.shape[-1]):
            mask = masks[:, :, i]
            points = get_contour(mask)
            score = res[0]['scores'][i]
            point_sets.append((list(points), score))
        print("Contours extracted")

        rectangularized_outlines = []
        if do_rectangularization:
            point_sets = list(map(lambda point_set_with_score: (rectangularize(point_set_with_score[0]), point_set_with_score[1]), point_sets))

        point_sets_mapped = []
        col, row = tile
        for points, score in point_sets:
            pp = list(map(lambda p: (p[0]+col*256, p[1]+row*256), points))
            if pp:
                point_sets_mapped.append((pp, score))
        point_sets = point_sets_mapped

        if not extent:
            rectangularized_outlines = point_sets
        else:
            for o, score in point_sets:
                georeffed = georeference(o, extent)
                if georeffed:
                    rectangularized_outlines.append((georeffed, score))
        return rectangularized_outlines

    def predict_path(self, img_path: str, extent=None, verbose=1) -> List[List[Tuple[int, int]]]:
        img = Image.open(img_path)
        data = np.asarray(img, dtype="uint8")
        return self.predict_array(img_data=data, extent=extent, verbose=verbose)


def test_images(annotations_file_name="predictions.json", processed_images_name="tested_images.txt", nr_images=None, target_dir=TEST_DATA_DIR):
    predictor = Predictor(os.path.join(os.getcwd(), "model", "stage2.h5"))
    annotations_path = os.path.join(os.getcwd(), annotations_file_name)
    images = glob.glob(os.path.join(target_dir, "**/*.jpg"), recursive=True)
    if nr_images:
        random.shuffle(images)
        images = images[:nr_images]
    annotations = []
    if os.path.isfile(annotations_path):
        with open(annotations_path, 'r', encoding="utf-8") as f:
            data = f.read()
            if data:
                annotations = json.loads(data)
    progress = 0
    nr_images = float(len(images))
    processed_images_path = os.path.join(os.getcwd(), processed_images_name)
    processed_images = []
    if os.path.isfile(processed_images_path):
        with open(processed_images_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            processed_images = list(map(lambda l: l[:-1], lines))  # remove '\n'

    for idx, img_path in enumerate(images):
        new_progress = 100*idx / nr_images
        if new_progress > progress:
            progress = new_progress
            print("Progress: {}%".format(progress))
            sys.stdout.flush()
        if img_path in processed_images:
            continue

        with open(processed_images_path, 'a') as f:
            f.write("{}\n".format(img_path))

        point_sets_with_score = [([], 0)]
        try:
            point_sets_with_score = predictor.predict_path(img_path, verbose=0)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("An error occured: " + str(e))

        for contour, score in point_sets_with_score:
            xs = list(map(lambda pt: int(pt[0]), contour))
            ys = list(map(lambda pt: int(pt[1]), contour))
            if contour:
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            else:
                bbox = []
            points_sequence = []
            for p in contour:
                points_sequence.append(int(round(p[0])))
                points_sequence.append(int(round(p[1])))
            ann = {
                "image_id": int(os.path.basename(img_path).replace(".jpg", "")),
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
    test_images()
