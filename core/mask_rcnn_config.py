from mask_rcnn.config import Config
import sys
from mask_rcnn import utils
from core.training_data import get_instances, get_instances_from_array
from core.settings import IMAGE_WIDTH
from typing import Tuple
import os
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as cocomask

osm_class_ids = {
    'building': 1
}

TRAINING_DATA_DIR = "/training-data"
VALIDATION_DATA_DIR = "/validation-data"
TEST_DATA_DIR = "/test-data"


class MyMaskRcnnConfig(Config):
    NAME = "osm"

    NUM_CLASSES = 2  # building & not building

    # Batch size is (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    LEARNING_RATE = 0.001

    # faster training
    STEPS_PER_EPOCH = 1000 // IMAGES_PER_GPU
    # STEPS_PER_EPOCH = 280741 // IMAGES_PER_GPU

    # all images
    # STEPS_PER_EPOCH = 1000

    # Each tile is 256 pixels across, training data is 3x3 tiles
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320

    USE_MINI_MASK = True
    # MINI_MASK_SHAPE = (128, 128)
    # MASK_SHAPE = (IMAGE_MIN_DIM, IMAGE_MIN_DIM)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 64
    # DETECTION_MAX_INSTANCES = 64

    VALIDATION_STEPS = 100  # 60317


class OsmMappingDataset(utils.Dataset):

    def __init__(self):
        utils.Dataset.__init__(self)
        print("Dataset: OsmMappingDataset")

    def load(self, images):
        self.add_class("osm", 0, "building")
        print("")
        print("Loading {} images...".format(len(images)))
        for idx, path in enumerate(images):
            self.add_image(source="osm", image_id=idx, path=path)
        print("Loaded.")

    def _get_image(self, path: str) -> np.ndarray:
        # img = Image.open(path)
        # data = np.asarray(img, dtype="uint8")
        data = cv2.imread(path)
        # print(data.shape)
        return data

    def _get_mask(self, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # images have fileextension ".tiff", masks have ".tif"
        assert not mask_path.endswith(".tiff")
        if not os.path.isfile(mask_path):
            raise RuntimeError("Mask does not exist")

        instances = get_instances(mask_path)
        class_ids = np.zeros(len(instances), np.int32)

        mask = np.zeros([IMAGE_WIDTH, IMAGE_WIDTH, len(instances)], dtype=np.uint8)
        for i, inst in enumerate(instances):
            class_ids[i] = osm_class_ids["building"]
            mask[:, :, i] = inst
        return mask, class_ids

    def load_image(self, image_id: str) -> np.ndarray:
        info = self.image_info[image_id]
        path = info["path"]
        return self._get_image(path)

    def load_mask(self, image_id: str) -> Tuple[np.ndarray, np.ndarray]:
        info = self.image_info[image_id]
        image_path = info["path"]
        return self._get_mask(image_path[:-1])


class InMemoryDataset(OsmMappingDataset):
    def __init__(self, path, limit=None, annotation_filename="annotation.json"):
        OsmMappingDataset.__init__(self)
        self._cache = {}
        print("Dataset: InMemoryDataset")
        self.no_logging = False
        self.path = path
        self.coco = COCO(os.path.join(path, annotation_filename))
        self.limit = limit
        self.coco_image_ids = None

    def get_mask_from_annotation(self, img):
        annotation_ids = self.coco.getAnnIds(imgIds=img['id'])
        annotations = self.coco.loadAnns(annotation_ids)
        # all_instances = np.zeros((img['height'], img['width']), dtype=np.uint8)
        # print("nr annotations: ", len(annotations))

        class_ids = np.zeros(len(annotations), np.int32)
        # print("Nr instances:", len(annotations))
        mask = np.zeros([IMAGE_WIDTH, IMAGE_WIDTH, len(annotations)], dtype=np.uint8)

        for i, ann in enumerate(annotations):
            rle = cocomask.frPyObjects(ann['segmentation'], img['height'], img['width'])
            m = cocomask.decode(rle)
            m = m.reshape((img['height'], img['width']))
            mask[:, :, i] = m
            class_ids[i] = osm_class_ids["building"]
        return mask, class_ids

    @staticmethod
    def get_mask_from_array(arr) -> Tuple[np.ndarray, np.ndarray]:
        instances = get_instances_from_array(arr)
        class_ids = np.zeros(len(instances), np.int32)
        # print("Nr instances:", len(instances))
        mask = np.zeros([IMAGE_WIDTH, IMAGE_WIDTH, len(instances)], dtype=np.uint8)
        for i, inst in enumerate(instances):
            class_ids[i] = osm_class_ids["building"]
            mask[:, :, i] = inst
        return mask, class_ids

    def load(self):
        image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())
        self.coco_image_ids = image_ids
        images = self.coco.loadImgs(image_ids)

        self.add_class("osm", 0, "building")
        print("")
        print("Loading {} images...".format(len(images)))
        progress = 0
        total_nr_images = len(images)
        for idx, coco_img in enumerate(images):
            if self.limit and idx >= self.limit:
                break

            image_path = os.path.join(self.path, "images", coco_img["file_name"])
            img = self._get_image(path=image_path)
            msk = self.get_mask_from_annotation(coco_img)
            self.add_image(source="osm", image_id=image_path, path=image_path)
            self._cache[image_path] = {
                "img": img,
                "mask": msk
            }

            new_progress = int(round(idx / total_nr_images * 100))
            if not self.no_logging and new_progress != progress:
                progress = new_progress
                print("Caching progress: {}% ({} images)".format(progress, idx+1))
                sys.stdout.flush()
        print("Loaded.")

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image_path = info["path"]
        return self._cache[image_path]["img"]

    def load_mask(self, image_id: str) -> Tuple[np.ndarray, np.ndarray]:
        info = self.image_info[image_id]
        image_path = info["path"]
        return self._cache[image_path]["mask"]
