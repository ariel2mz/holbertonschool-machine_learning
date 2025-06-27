#!/usr/bin/env python3
"""asdfgfdsasdf"""
import tensorflow as tf
import numpy as np
K = tf.keras


class Yolo:
    """asdsadasdasds jkjkjkjka"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        sadasdasdsa

        Parameters:
        - model_path: sada
        - classes_path: adsadas
        - class_t: asdasds
        - nms_t: sadasds
        - anchors:asdasdasda
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs with deterministic operations.
        Args:
            adsadsdsa
        Returns:
            Tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_h, image_w = image_size
        input_h, input_w = self.model.input.shape[1:3]

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_anchors, _ = output.shape

            grid_y = np.arange(grid_h).reshape(grid_h, 1, 1, 1)
            grid_x = np.arange(grid_w).reshape(1, grid_w, 1, 1)

            tx = output[..., 0:1]
            ty = output[..., 1:2]
            tw = output[..., 2:3]
            th = output[..., 3:4]
            box_confidence = output[..., 4:5]
            class_probs = output[..., 5:]

            anchor_w = self.anchors[i, :, 0].reshape(1, 1, num_anchors, 1)
            anchor_h = self.anchors[i, :, 1].reshape(1, 1, num_anchors, 1)

            bx = (1 / (1 + np.exp(-tx)) + grid_x) / grid_w
            by = (1 / (1 + np.exp(-ty)) + grid_y) / grid_h
            bw = (np.exp(tw) * anchor_w) / input_w
            bh = (np.exp(th) * anchor_h) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.concatenate((x1, y1, x2, y2), axis=-1)

            boxes.append(box)
            box_confidences.append(1 / (1 + np.exp(-box_confidence)))
            box_class_probs.append(1 / (1 + np.exp(-class_probs)))

        return boxes, box_confidences, box_class_probs