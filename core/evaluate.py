import os
from core.mask_rcnn_config import VALIDATION_DATA_DIR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from core.predict import test_images


def evaluate():
    ground_truth_annotations = COCO(os.path.join(VALIDATION_DATA_DIR, "annotation.json"))
    predictions_path = os.path.join(os.getcwd(), "eval_predictions.json")
    submission_file = json.loads(open(predictions_path).read())
    results = ground_truth_annotations.loadRes(submission_file)
    cocoEval = COCOeval(ground_truth_annotations, results, 'segm')
    cocoEval.evaluate()
    cocoEval.accumulate()
    average_precision = cocoEval._summarize(ap=1, iouThr=0.5, areaRng="all", maxDets=100)
    average_recall = cocoEval._summarize(ap=0, iouThr=0.5, areaRng="all", maxDets=100)
    print("Average Precision : {} || Average Recall : {}".format(average_precision, average_recall))


if __name__ == "__main__":
    test_images("eval_predictions.json", "eval_tested_images.txt", 10, VALIDATION_DATA_DIR)
    evaluate()
