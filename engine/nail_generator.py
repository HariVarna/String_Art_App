import numpy as np
import cv2
import math

DEFAULT_MARGIN_RATIO = 0.0625


def generate_circle_nails(image_size, nail_count, margin_ratio=DEFAULT_MARGIN_RATIO):
    center = (image_size - 1) / 2.0
    margin = max(2, int(round(image_size * margin_ratio)))
    radius = center - margin

    nails = []

    for i in range(nail_count):
        angle = 2 * math.pi * i / nail_count
        x = int(round(center + radius * math.cos(angle)))
        y = int(round(center + radius * math.sin(angle)))
        nails.append((x, y))

    return nails


def draw_nails(image_size, nails):
    canvas = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

    for i, (x, y) in enumerate(nails):
        cv2.circle(canvas, (x, y), 2, (0, 0, 0), -1)

    return canvas # Draw nails on the canvas
