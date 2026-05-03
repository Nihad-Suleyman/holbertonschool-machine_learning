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
        """Process the outputs from the Darknet model"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_h = image_size[0]
        image_w = image_size[1]

        for i, output in enumerate(outputs):
            grid_h = output.shape[0]
            grid_w = output.shape[1]
            anchor_boxes = output.shape[2]

            box = output[..., 0:4].copy()

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            col = np.arange(grid_w).reshape(1, grid_w, 1)
            row = np.arange(grid_h).reshape(grid_h, 1, 1)

            box[..., 0] = (1 / (1 + np.exp(-box[..., 0])) + col) / grid_w
            box[..., 1] = (1 / (1 + np.exp(-box[..., 1])) + row) / grid_h

            box[..., 2] = (
                np.exp(box[..., 2]) * self.anchors[i, :, 0]
            ) / input_w
            box[..., 3] = (
                np.exp(box[..., 3]) * self.anchors[i, :, 1]
            ) / input_h

            x1 = (box[..., 0] - box[..., 2] / 2) * image_w
            y1 = (box[..., 1] - box[..., 3] / 2) * image_h
            x2 = (box[..., 0] + box[..., 2] / 2) * image_w
            y2 = (box[..., 1] + box[..., 3] / 2) * image_h

            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
