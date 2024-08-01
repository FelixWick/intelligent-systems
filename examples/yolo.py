from ultralytics import YOLO

detection_model = YOLO("yolov8n.pt")
detection_model.predict("https://ultralytics.com/images/bus.jpg", save=True, conf=0.5, show_boxes=True, show_labels=True, show_conf=True)

segmentation_model = YOLO("yolov8n-seg.pt")
segmentation_model.predict("https://ultralytics.com/images/bus.jpg", save=True, conf=0.5, show_boxes=False, show_labels=False, show_conf=False)
