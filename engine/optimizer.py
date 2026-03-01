import cv2
import numpy as np

def pick_best_nail(target, canvas, nails, current):
    best_score = -1
    best_index = current

    for i in range(len(nails)):
        if i == current:
            continue

        temp = canvas.copy()
        cv2.line(temp, nails[current], nails[i], 0, 1)

        before = np.mean(np.abs(target - canvas))
        after = np.mean(np.abs(target - temp))

        score = before - after

        if score > best_score:
            best_score = score
            best_index = i

    return best_index
