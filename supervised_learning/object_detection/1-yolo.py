#!/usr/bin/env python3
"""A module that does the trick"""
from tensorflow import keras as K
import numpy as np


class Yolo:
    """A class that does the trick"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializer for the class"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process the outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Extract components
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            confidence = self.sigmoid(output[..., 4])[..., np.newaxis]
            class_probs = self.sigmoid(output[..., 5:])

            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            cx, cy = np.meshgrid(cx, cy)

            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            bx = (self.sigmoid(t_x) + cx) / grid_w
            by = (self.sigmoid(t_y) + cy) / grid_h

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            anchor_w = anchor_w.reshape((1, 1, anchor_boxes))
            anchor_h = anchor_h.reshape((1, 1, anchor_boxes))

            bw = (np.exp(t_w) * anchor_w) / self.model.input.shape[1]
            bh = (np.exp(t_h) * anchor_h) / self.model.input.shape[2]

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)

            boxes.append(box)
            box_confidences.append(confidence)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
