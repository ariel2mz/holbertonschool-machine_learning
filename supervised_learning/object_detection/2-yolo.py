#!/usr/bin/env python3
"""asdfgfdsasdf"""
import tensorflow as tf
import numpy as np
K = tf.keras


class Yolo:
    """YOLO v3 object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        sadsadsadsad sadsadsadsad asdsa
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        _, self.input_width, self.input_height, _ = self.model.input.shape

    def process_outputs(self, outputs, image_size):
        """
        sadsadsadsa sadsad
        """
        bxs, bConf, bClassProb = [], [], []
        imgH, imgW = image_size

        for i, out in enumerate(outputs):
            gh, gw, an, _ = out.shape
            gy = np.arange(gh).reshape(gh, 1, 1, 1)
            gx = np.arange(gw).reshape(1, gw, 1, 1)

            tx, ty = out[..., 0:1], out[..., 1:2]
            tw, th = out[..., 2:3], out[..., 3:4]
            conf = out[..., 4:5]
            probs = out[..., 5:]

            aw = self.anchors[i, :, 0].reshape(1, 1, an, 1)
            ah = self.anchors[i, :, 1].reshape(1, 1, an, 1)

            bx = (1 / (1 + np.exp(-tx)) + gx) / gw
            by = (1 / (1 + np.exp(-ty)) + gy) / gh
            bw = (np.exp(tw) * aw) / self.input_width
            bh = (np.exp(th) * ah) / self.input_height

            x1 = (bx - bw / 2) * imgW
            y1 = (by - bh / 2) * imgH
            x2 = (bx + bw / 2) * imgW
            y2 = (by + bh / 2) * imgH
            b = np.concatenate((x1, y1, x2, y2), axis=-1)

            bxs.append(b)
            bConf.append(1 / (1 + np.exp(-conf)))
            bClassProb.append(1 / (1 + np.exp(-probs)))

        return bxs, bConf, bClassProb

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        sadsadsadsa
        asdsadssa
        """
        fBxs, bCls, bScrs = [], [], []

        for i in range(len(boxes)):
            conf = box_confidences[i]
            probs = box_class_probs[i]
            scores = conf * probs

            cls = np.argmax(scores, axis=-1)
            maxScr = np.max(scores, axis=-1)
            mask = maxScr >= self.class_t

            fBxs.append(boxes[i][mask])
            bCls.append(cls[mask])
            bScrs.append(maxScr[mask])

        fBxs = np.concatenate(fBxs, axis=0)
        bCls = np.concatenate(bCls, axis=0)
        bScrs = np.concatenate(bScrs, axis=0)

        return fBxs, bCls, bScrs
