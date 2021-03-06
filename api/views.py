import sys
import os
import math
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from django.http import JsonResponse
from .serializers import InferenceRequestSerializer, InferenceRequest
from core.utils import georeference, rectangularize
from core.predict import Predictor
import base64
import numpy as np
from PIL import Image
import io
from shapely import geometry, wkt
import geojson
import traceback
import glob

model_path = r"D:\_models\mask_rcnn_osm_0100.h5"
if not os.path.isfile(model_path):
    models = glob.glob(os.path.join("/model", "**/*.h5"), recursive=True)
    if not models:
        raise RuntimeError("No models were found in the '/model' folder")
    else:
        model_path = models[0]
_predictor = Predictor(model_path)


"""
Request format (url: localhost:8000/inference):
{
    "bbox": {
        "lat_min": 12,
        "lat_max": 12,
        "lon_min": 12,
        "lon_max": 12
    },
    "image_data": "123"
}
"""


def diff(a, b, check_intersection=True, check_containment=False, min_area=20):
    """
     * Returns a representative point for each feature from a, that has no intersecting feature in b
    :param min_area:
    :param check_containment:
    :param check_intersection:
    :param a:
    :param b:
    :return:
    """

    res = []
    for feature_a, class_name_a in a:
        if not feature_a.area >= min_area:
            continue

        hit = False
        for feature_b, class_name_b in b:
            if not feature_b.area >= min_area:
                continue

            if (check_intersection and feature_b.intersects(feature_a)) \
                    or (check_containment and feature_b.within(feature_a)):
                hit = True
                break
        if (check_intersection and not hit) or (check_containment and hit):
            res.append((feature_a, class_name_a))
    return res


def to_final_geojson(features, props, add_predicted_class_to_props=False, to_point=False):
    res = []
    if not props and add_predicted_class_to_props:
        props = {}
    for f, class_name in features:
        if add_predicted_class_to_props:
            props['class'] = class_name
        # p = f
        if not f.is_valid:
            continue
        if to_point:
            p = f.representative_point().buffer(4)
        else:
            p = f
        res.append(to_geojson(p, properties=props))
    return res


@api_view(['GET', 'POST'])
def request_inference(request):
    if request.method == "GET":
        return JsonResponse({'hello': 'world'})
    else:
        data = JSONParser().parse(request)
        inference_serializer = InferenceRequestSerializer(data=data)
        if not inference_serializer.is_valid():
            print("Errors: ", inference_serializer.errors)
            return JsonResponse({'errors': inference_serializer.errors})

        inference = InferenceRequest(**inference_serializer.data)
        try:
            res = _predict(inference)
            # coll = "GEOMETRYCOLLECTION({})".format(", ".join(res))
            # with open(r"D:\training_images\_last_predicted\wkt.txt", 'w') as f:
            #     f.write(coll)

            ref_features = list(map(lambda f: (wkt.loads(f), 'reference'), inference.reference_features))

            original = list(res)

            deleted_features = diff(ref_features, res)
            added_features = diff(res, ref_features)
            changed_features = diff(res, ref_features, check_intersection=False, check_containment=True)
            print("Deleted: ", len(deleted_features))
            print("Added: ", len(added_features))
            print("Changed: ", len(changed_features))
            print("Done")

            output = {
                'features': list(map(lambda feat: to_geojson(geom=feat[0], properties={'type': feat[1], 'area': feat[0].area}), original)),
                'deleted': to_final_geojson(deleted_features, {'type': 'deleted'}, to_point=True),
                'added': to_final_geojson(added_features, {'type': 'added'}, True),
                'changed': to_final_geojson(changed_features, {'type': 'changed'})
            }

            return JsonResponse(output)
        except Exception as e:
            tb = ""
            if traceback:
                tb = traceback.format_exc()
            print("Server error: {}, {}", sys.exc_info(), tb)
            msg = str(e)
            return JsonResponse({'error': msg})


def _predict(request: InferenceRequest):
    print("Decoding image")
    b64 = base64.b64decode(request.image_data)
    print("Image decoded")
    barr = io.BytesIO(b64)
    img = Image.open(barr)
    img = img.convert("RGB")
    width, height = img.size
    print("Received image size: ", img.size)
    extent = {
        'x_min': request.x_min,
        'y_min': request.y_min,
        'x_max': request.x_max,
        'y_max': request.y_max,
        'img_width': width,
        'img_height': height
    }

    img_size = 512  # the image will be cropped for prediction
    scale_by_factor = 3  # and scaled by this factor for improved accuracy
    new_width = width * scale_by_factor
    new_height = height * scale_by_factor

    img = img.resize((new_width, new_height), Image.ANTIALIAS)

    all_polygons = []
    cols = int(math.ceil(new_width / float(img_size)))
    rows = int(math.ceil(new_height / float(img_size)))
    images_to_predict = []
    tiles_by_img_id = {}
    count = 0
    for col in range(0, cols):
        for row in range(0, rows):
            count += 1
            print("Processing tile (x={},y={})".format(col, row))
            start_width = col * img_size
            start_height = row * img_size
            img_copy = img.crop((start_width, start_height, start_width+img_size, start_height+img_size))
            print("Cropped image size: ", img_copy.size)
            # img_copy = img.resize((512, 512), Image.ANTIALIAS)
            # img_copy.save(r"C:\Users\Martin\AppData\Local\Temp\deep_osm\cropped_{}.png".format(str(count)))
            arr = np.asarray(img_copy)
            img_id = "img_id_{}_{}".format(col, row)
            tiles_by_img_id[img_id] = (col, row)
            images_to_predict.append((arr, img_id))
            # break
        # break
    # images_to_predict = images_to_predict[0:3]
    point_sets = _predictor.predict_arrays(images=images_to_predict)

    count = 0
    for points, img_id, class_name in point_sets:
        count += 1
        col, row = tiles_by_img_id[img_id]
        points = list(map(lambda p: ((p[0]+col*img_size)/scale_by_factor, (p[1]+row*img_size)/scale_by_factor), points))
        if request.rectangularize:
            print("Rectangularizing point set {}/{}...".format(count, len(point_sets)))
            points = rectangularize(points)
            print("Rectangularizing complete")
        print("Georeferencing point set {}/{}...".format(count, len(point_sets)))
        georeffed = georeference(points, extent)
        print("Georeferencing complete")
        if georeffed:
            points = georeffed
        polygon = geometry.Polygon(points)
        all_polygons.append((polygon, class_name))

    return all_polygons


def to_geojson(geom, properties=None):
    props = {}
    if properties:
        props = properties
    f = geojson.Feature(geometry=geom, properties=props)
    return f
