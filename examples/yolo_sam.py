import sys

import numpy as np

from ultralytics import YOLO

import cv2

from segment_anything import sam_model_registry
from segment_anything import SamPredictor

from IPython import embed


def draw_bounding_boxes(image, result):
    for box in result.boxes:
        cords = box.xyxy[0].tolist()
        start_point = (int(cords[0]), int(cords[1]))
        end_point = (int(cords[2]), int(cords[3]))
        cv2.rectangle(image, start_point, end_point, color=(0, 255, 0), thickness=2)
        class_id = result.names[box.cls[0].item()]
        cv2.putText(
            image,
            class_id,
            (int(cords[0]), int(cords[1]) - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            thickness = 2,
            lineType=cv2.LINE_AA
            )
        conf = round(box.conf[0].item(), 2)
        cv2.putText(
            image,
            str(conf),
            (int(cords[0]), int(cords[3]) - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            thickness = 2,
            lineType=cv2.LINE_AA
            )
        print("Object type:", class_id)
        print("Probability:", conf)
    cv2.imwrite("example_with_bounding_boxes.jpg", image)


def draw_mask(image, mask_generated) :
  masked_image = image.copy()
  masked_image = np.where(mask_generated.astype(int), np.array([0,255,0], dtype='uint8'), masked_image)
  masked_image = masked_image.astype(np.uint8)
  return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)


def main(args):
    model = YOLO("yolov8m.pt")

    image = cv2.imread("cat_dog.jpg")
    results = model.predict(image)
    result = results[0]
    print("number of detected objects: {}".format(len(result.boxes)))
    # print(result.names)

    bounded_image = image.copy()
    draw_bounding_boxes(bounded_image, result)

# git clone git@github.com:facebookresearch/segment-anything.git
# wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
    DEVICE = "cpu" #cpu,cuda
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image)

    boxes = np.array(results[0].to('cpu').boxes.data)
    mask, _, _ = mask_predictor.predict(box=boxes[1][:-2])
    mask = np.transpose(mask, (1, 2, 0))
    segmented_image = draw_mask(image, mask)
    cv2.imwrite("example_with_segmentation.jpg", segmented_image)

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
