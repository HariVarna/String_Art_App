import numpy as np
import cv2
import math

def generate_circle_nails(image_size, nail_count):
    center = image_size // 2
    radius = image_size // 2 - 20

    nails = []

    for i in range(nail_count):
        angle = 2 * math.pi * i / nail_count
        x = int(center + radius * math.cos(angle))
        y = int(center + radius * math.sin(angle))
        nails.append((x, y))

    return nails


def draw_nails(image_size, nails):
    canvas = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

    for i, (x, y) in enumerate(nails):
        cv2.circle(canvas, (x, y), 2, (0, 0, 0), -1)

    return canvas # Draw nails on the canvas
