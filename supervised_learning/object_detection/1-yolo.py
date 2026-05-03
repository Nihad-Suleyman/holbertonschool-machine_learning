#!/usr/bin/env python3
"""YOLO v3 object detection"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        img_h, img_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]

            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))

            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cy = np.arange(grid_h).reshape(grid_h, 1, 1)

            bx = (1 / (1 + np.exp(-t_xy[..., 0])) + cx) / grid_w
            by = (1 / (1 + np.exp(-t_xy[..., 1])) + cy) / grid_h

            anchors = self.anchors[i]
            bw = anchors[:, 0] * np.exp(t_wh[..., 0]) / input_w
            bh = anchors[:, 1] * np.exp(t_wh[..., 1]) / input_h

            x1 = (bx - bw / 2) * img_w
            y1 = (by - bh / 2) * img_h
            x2 = (bx + bw / 2) * img_w
            y2 = (by + bh / 2) * img_h

            box = np.zeros_like(output[..., 0:4])
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)
            box_confidences.append(confidence)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
