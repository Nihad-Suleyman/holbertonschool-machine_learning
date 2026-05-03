#!/usr/bin/env python3
"""YOLO v3 object detection"""
import tensorflow.keras as K


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor for Yolo class
        """
        self.model = K.models.load_model(model_path)
        self.class_names = []

        with open(classes_path, "r") as f:
            for line in f:
                self.class_names.append(line.strip())

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
