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
        for i, output in enumerate(outputs):
            grid_h = output.shape[0]
            grid_w = output.shape[1]
            anchor_boxes = output.shape[2]
            anchors = self.anchors[i]
            anchors = anchors.reshape((1, 1, anchor_boxes, 2))
            t_xywh = output[..., :4]
            t_x = t_xywh[..., 0]
            t_y = t_xywh[..., 1]
            t_w = t_xywh[..., 2]
            t_h = t_xywh[..., 3]

# 3B. Slice the box confidence (single channel)
box_confidence = output[..., 4:5]   # keep last axis size = 1 for broadcasting later

# 3C. Slice the class-probability vector (remaining channels)
class_probs = output[..., 5:]       # shape: (grid_h, grid_w, anchor_boxes, num_classes)