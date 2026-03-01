import cv2

def draw_line_on_nails(canvas, nails, start, end):
    p1 = nails[start]
    p2 = nails[end]
    cv2.line(canvas, p1, p2, 0, 1)
    return canvas
