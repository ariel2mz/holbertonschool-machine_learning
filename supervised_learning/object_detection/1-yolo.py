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
            outputs: List of numpy.ndarrays containing predictions from Darknet
                    Each output has shape (grid_height, grid_width,
                                            anchor_boxes, 4 + 1 + classes)
            image_size: Original image size [height, width]
        Returns:
            Tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        boxconf = []  # Changed from 'conf' to avoid conflict
        boxprobs = []  # Changed from 'prob' to avoid potential conflicts
        image_h, image_w = image_size

        # Get model input shape from the model itself
        self.input_height, self.input_width = self.model.input.shape[1:3]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchors, _ = output.shape
            
            # Create grid coordinates
            grid_y = np.arange(grid_h).reshape(grid_h, 1, 1)
            grid_x = np.arange(grid_w).reshape(1, grid_w, 1)

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]
            conf = output[..., 4]  # Renamed to avoid conflict
            probs = output[..., 5:]  # Renamed for clarity

            # Reshape anchors to match output dimensions
            anchor_w = self.anchors[i, :, 0].reshape(1, 1, anchors)
            anchor_h = self.anchors[i, :, 1].reshape(1, 1, anchors)

            # Calculate box coordinates
            bx = (1 / (1 + np.exp(-tx)) + grid_x) / grid_w
            by = (1 / (1 + np.exp(-ty)) + grid_y) / grid_h
            bw = (np.exp(tw) * anchor_w) / self.input_width
            bh = (np.exp(th) * anchor_h) / self.input_height

            # Convert to image coordinates
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            # Stack coordinates
            box = np.concatenate(
                [x1[..., np.newaxis], y1[..., np.newaxis], 
                x2[..., np.newaxis], y2[..., np.newaxis]], axis=-1
            )
            boxes.append(box)
            
            # Apply sigmoid and store results
            boxconf.append(1 / (1 + np.exp(-conf)))
            boxprobs.append(1 / (1 + np.exp(-probs)))

        return boxes, boxconf, boxprobs