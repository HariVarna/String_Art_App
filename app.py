from collections import deque
from functools import lru_cache
import base64
import binascii
import os
from pathlib import Path
from time import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, url_for

from engine.nail_generator import generate_circle_nails
from engine.optimizer import apply_line_to_residual, build_line_cache, pick_best_nail

app = Flask(__name__)

OUTPUT_FOLDER = "static/outputs"

COMPUTE_SIZE = 320
OUTPUT_SIZE = 1800
NAIL_COUNT = 320
MAX_LINES = 5000
DEFAULT_SELECTED_LINES = 2500
MIN_LINE_SCORE = 3.0
MIN_NAIL_GAP = 8
LINE_WEIGHT = 12.0
THREAD_THICKNESS = 1
CIRCLE_MARGIN = 10
THREAD_OPACITY = 0.08
PORTRAIT_FACE_FILL = 0.42
PORTRAIT_CROP_SCALE = 1.95
PORTRAIT_VERTICAL_SHIFT = 0.08

app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

LAST_JOB = {
    "steps": [],
    "available_lines": 0,
    "selected_lines": DEFAULT_SELECTED_LINES,
    "version": None,
}


def ensure_directories():
    Path(app.config["OUTPUT_FOLDER"]).mkdir(parents=True, exist_ok=True)


def center_crop_square(image):
    height, width = image.shape[:2]
    side = min(height, width)
    y = (height - side) // 2
    x = (width - side) // 2
    return image[y : y + side, x : x + side]


def crop_square_from_center(image, center_x, center_y, side_length):
    height, width = image.shape[:2]
    side_length = int(round(min(side_length, height, width)))
    half = side_length / 2.0

    left = int(round(center_x - half))
    top = int(round(center_y - half))
    right = left + side_length
    bottom = top + side_length

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height

    left = max(0, left)
    top = max(0, top)

    return image[top:bottom, left:right]


def auto_focus_portrait(image):
    base_crop = center_crop_square(image)

    if FACE_CASCADE.empty():
        return base_crop

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_face_size = max(80, min(image.shape[:2]) // 7)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )

    if len(faces) == 0:
        return base_crop

    x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
    face_fill = max(width, height) / float(min(image.shape[:2]))
    if face_fill >= PORTRAIT_FACE_FILL:
        return base_crop

    center_x = x + width / 2.0
    center_y = y + height / 2.0 + height * PORTRAIT_VERTICAL_SHIFT
    side_length = max(width, height) * PORTRAIT_CROP_SCALE
    focused = crop_square_from_center(image, center_x, center_y, side_length)
    return center_crop_square(focused)


def create_circle_mask(size, margin):
    center = (size - 1) / 2.0
    radius = center - margin
    yy, xx = np.ogrid[:size, :size]
    return (xx - center) ** 2 + (yy - center) ** 2 <= radius**2


def clamp_line_count(value, upper_limit):
    if upper_limit <= 0:
        return 0

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_SELECTED_LINES

    return max(1, min(parsed, upper_limit, MAX_LINES))


def decode_uploaded_file(file_storage):
    image_bytes = file_storage.read()
    if not image_bytes:
        raise ValueError("Please upload an image file.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read the uploaded image.")

    return image


def decode_prepared_image(data_url):
    try:
        _, encoded = data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
    except (ValueError, binascii.Error):
        raise ValueError("Could not read the cropped image preview.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read the cropped image preview.")

    return image


def load_input_image(file_storage, prepared_image_data):
    if prepared_image_data:
        return decode_prepared_image(prepared_image_data)

    if not file_storage or not file_storage.filename:
        raise ValueError("Please upload an image file.")

    return decode_uploaded_file(file_storage)


def preprocess_image(image):
    image = auto_focus_portrait(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (COMPUTE_SIZE, COMPUTE_SIZE), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    soft = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, 1.35, soft, -0.35, 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    circle_mask = create_circle_mask(COMPUTE_SIZE, CIRCLE_MARGIN)
    gray = np.where(circle_mask, gray, 255).astype(np.uint8)

    return gray


@lru_cache(maxsize=4)
def get_generation_assets(compute_size, output_size, nail_count):
    nails = generate_circle_nails(compute_size, nail_count)
    big_nails = generate_circle_nails(output_size, nail_count)
    line_cache = build_line_cache(nails, compute_size)
    return nails, big_nails, line_cache


def write_steps_file(steps, line_count):
    steps_path = os.path.join(app.config["OUTPUT_FOLDER"], "steps.txt")

    with open(steps_path, "w", encoding="utf-8") as handle:
        handle.write(format_steps_text(steps, line_count))
        handle.write("\n")


def build_nail_sequence(steps, line_count):
    limited_steps = steps[:line_count]
    if not limited_steps:
        return []

    sequence = [limited_steps[0][0], limited_steps[0][1]]
    for _, end_index in limited_steps[1:]:
        sequence.append(end_index)

    return sequence


def format_steps_text(steps, line_count):
    return ", ".join(str(nail) for nail in build_nail_sequence(steps, line_count))


def render_line_art(steps, line_count):
    nails, _, _ = get_generation_assets(
        COMPUTE_SIZE,
        OUTPUT_SIZE,
        NAIL_COUNT,
    )

    canvas = np.ones((COMPUTE_SIZE, COMPUTE_SIZE), dtype=np.float32)
    line_mask = np.zeros((COMPUTE_SIZE, COMPUTE_SIZE), dtype=np.uint8)

    for start_index, end_index in steps[:line_count]:
        line_mask.fill(0)
        cv2.line(
            line_mask,
            nails[start_index],
            nails[end_index],
            255,
            THREAD_THICKNESS,
            lineType=cv2.LINE_AA,
        )
        coverage = (line_mask.astype(np.float32) / 255.0) * THREAD_OPACITY
        canvas *= 1.0 - coverage

    rendered = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(
        rendered,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )


def write_preview_output(steps, selected_lines):
    line_output_path = os.path.join(app.config["OUTPUT_FOLDER"], "line.png")
    rendered = render_line_art(steps, selected_lines)
    cv2.imwrite(line_output_path, rendered)
    write_steps_file(steps, selected_lines)
    return int(time() * 1000)


def build_page_context(**extra):
    context = {
        "requested_lines": LAST_JOB["selected_lines"] or DEFAULT_SELECTED_LINES,
    }

    if LAST_JOB["available_lines"]:
        context.update(
            {
                "output": "outputs/gray.png",
                "line": "outputs/line.png",
                "steps": "outputs/steps.txt",
                "steps_text": format_steps_text(
                    LAST_JOB["steps"],
                    LAST_JOB["selected_lines"],
                ),
                "available_lines": LAST_JOB["available_lines"],
                "selected_lines": LAST_JOB["selected_lines"],
                "version": LAST_JOB["version"],
            }
        )

    context.update(extra)
    return context


def generate_string_art(image, selected_lines):
    ensure_directories()

    gray = preprocess_image(image)
    gray_output_path = os.path.join(app.config["OUTPUT_FOLDER"], "gray.png")
    cv2.imwrite(gray_output_path, gray)

    nails, _, line_cache = get_generation_assets(
        COMPUTE_SIZE,
        OUTPUT_SIZE,
        NAIL_COUNT,
    )

    residual = (255 - gray).astype(np.float32).reshape(-1)
    steps = []
    current = 0
    recent_nails = deque(maxlen=24)

    for line_number in range(MAX_LINES):
        next_nail, score = pick_best_nail(
            residual,
            line_cache,
            current,
            min_gap=MIN_NAIL_GAP,
            recent_nails=tuple(recent_nails),
        )

        if next_nail == current or score < MIN_LINE_SCORE:
            break

        line_pixels = line_cache[current][next_nail]
        apply_line_to_residual(residual, line_pixels, LINE_WEIGHT)
        steps.append((current, next_nail))
        recent_nails.append(next_nail)
        current = next_nail

        if line_number % 500 == 0:
            print(line_number, "/", MAX_LINES)

    if not steps:
        raise ValueError("This image did not produce any usable string-art lines.")

    available_lines = len(steps)
    selected_lines = clamp_line_count(selected_lines, available_lines)
    version = write_preview_output(steps, selected_lines)

    LAST_JOB["steps"] = steps
    LAST_JOB["available_lines"] = available_lines
    LAST_JOB["selected_lines"] = selected_lines
    LAST_JOB["version"] = version

    return {
        "available_lines": available_lines,
        "selected_lines": selected_lines,
        "version": version,
    }


@app.route("/preview", methods=["POST"])
def preview():
    if not LAST_JOB["steps"]:
        return jsonify({"error": "Generate an art piece first."}), 400

    payload = request.get_json(silent=True) or {}
    selected_lines = clamp_line_count(
        payload.get("line_count"),
        LAST_JOB["available_lines"],
    )
    version = write_preview_output(LAST_JOB["steps"], selected_lines)

    LAST_JOB["selected_lines"] = selected_lines
    LAST_JOB["version"] = version

    return jsonify(
        {
            "image_url": url_for("static", filename="outputs/line.png", v=version),
            "steps_url": url_for("static", filename="outputs/steps.txt", v=version),
            "steps_text": format_steps_text(LAST_JOB["steps"], selected_lines),
            "selected_lines": selected_lines,
            "available_lines": LAST_JOB["available_lines"],
            "version": version,
        }
    )


@app.route("/", methods=["GET", "POST"])
def index():
    ensure_directories()

    if request.method == "POST":
        file = request.files.get("image")
        prepared_image_data = request.form.get("prepared_image", "")
        requested_lines = request.form.get("requested_lines", DEFAULT_SELECTED_LINES)

        try:
            image = load_input_image(file, prepared_image_data)
            result = generate_string_art(image, requested_lines)
        except ValueError as exc:
            return render_template(
                "index.html",
                **build_page_context(
                    error=str(exc),
                    requested_lines=clamp_line_count(requested_lines, MAX_LINES),
                ),
            )

        return render_template(
            "index.html",
            **build_page_context(
                available_lines=result["available_lines"],
                selected_lines=result["selected_lines"],
                version=result["version"],
                requested_lines=result["selected_lines"],
            ),
        )

    return render_template("index.html", **build_page_context())


if __name__ == "__main__":
    app.run(debug=True)
