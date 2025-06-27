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
        asdasdssadsa
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes = output.shape[:3]
            anchors = self.anchors[i].reshape((1, 1, anchor_boxes, 2))

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))

            col = np.tile(np.arange(0, grid_w), grid_h).reshape(grid_w, grid_h).T
            row = np.tile(np.arange(0, grid_h), grid_w).reshape(grid_h, grid_w)

            col = col.reshape(grid_h, grid_w, 1)
            row = row.reshape(grid_h, grid_w, 1)

            bx = (1 / (1 + np.exp(-t_x)) + col) / grid_w
            by = (1 / (1 + np.exp(-t_y)) + row) / grid_h
            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]
            bw = anchors[..., 0] * np.exp(t_w) / input_w
            bh = anchors[..., 1] * np.exp(t_h) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(class_probs)

        return (boxes, box_confidences, box_class_probs)