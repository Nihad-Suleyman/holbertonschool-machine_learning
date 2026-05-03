#!/usr/bin/env python3
"""A module that does the trick"""
from tensorflow import keras as K
import numpy as np
import os
import cv2


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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes and confidence scores"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, conf, prob in zip(boxes,
                                 box_confidences,
                                 box_class_probs):
            scores = conf * prob
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])
        if len(filtered_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Non-max suppression"""
        final_boxes = []
        final_classes = []
        final_scores = []
        unique_classes = np.unique(box_classes)
        for clss in unique_classes:
            indexs = np.where(box_classes == clss)

            class_boxes = filtered_boxes[indexs]
            class_scores = box_scores[indexs]

            order = np.argsort(class_scores)[::-1]
            class_boxes = class_boxes[order]
            class_scores = class_scores[order]

            while len(class_boxes) > 0:
                best_box = class_boxes[0]
                best_score = class_scores[0]

                final_boxes.append(best_box)
                final_classes.append(clss)
                final_scores.append(best_score)

                if len(class_boxes) == 1:
                    break
                rest_boxes = class_boxes[1:]

                x1 = np.maximum(best_box[0], rest_boxes[:, 0])
                y1 = np.maximum(best_box[1], rest_boxes[:, 1])
                x2 = np.minimum(best_box[2], rest_boxes[:, 2])
                y2 = np.minimum(best_box[3], rest_boxes[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                area_best = (best_box[2] - best_box[0]) * (best_box[3]
                                                           - best_box[1])
                area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * \
                            (rest_boxes[:, 3] - rest_boxes[:, 1])

                union = area_best + area_rest - inter_area
                ious = inter_area / union

                keep = np.where(ious < self.nms_t)[0]
                class_boxes = rest_boxes[keep]
                class_scores = class_scores[1:][keep]

        return (
            np.array(final_boxes),
            np.array(final_classes),
            np.array(final_scores)
        )

    @staticmethod
    def load_images(folder_path):
        """Load images from folder"""
        images = []
        image_paths = []

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)

            if image is not None:
                images.append(image)
                image_paths.append(file_path)

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess images"""
        pimages = []
        shapes = []
        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]
        for img in images:
            img_shape = img.shape[0], img.shape[1]
            shapes.append(img_shape)
            image = cv2.resize(img, (input_w, input_h),
                               interpolation=cv2.INTER_CUBIC)
            image = image / 255
            pimages.append(image)
        pimages = np.array(pimages)
        image_shapes = np.array(shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Show boxes"""
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            label = '{} {:.2f}'.format(class_name, score)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
        cv2.imshow(file_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Predict on images"""
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        outputs = self.model.predict(pimages)
        predictions = []
        for i in range(len(images)):
            boxes, box_confidences, box_class_probs = self.process_outputs(
                [output[i] for output in outputs], image_shapes[i])
            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs)
            box_predictions, predicted_box_classes, predicted_box_scores = self.non_max_suppression(
                filtered_boxes, box_classes, box_scores)
            predictions.append((box_predictions, predicted_box_classes, predicted_box_scores))
        return predictions, image_paths
