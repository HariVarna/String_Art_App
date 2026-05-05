from collections import deque
from functools import lru_cache
import base64
import binascii
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
from time import time
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from engine.nail_generator import generate_circle_nails
from engine.optimizer import apply_line_to_residual, build_line_cache, pick_best_nail

app = Flask(__name__)

OUTPUT_FOLDER = "static/outputs"
SESSIONS_FOLDER = "static/sessions"
DATA_FOLDER = "data"
HISTORY_FILE = os.path.join(DATA_FOLDER, "sessions.json")
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

COMPUTE_SIZE = 420
OUTPUT_SIZE = 2400
PDF_OUTPUT_SIZE = 900
NAIL_COUNT = 360
MAX_LINES = 6500
DEFAULT_SELECTED_LINES = 3200
MIN_LINE_SCORE = 8.0
MIN_NAIL_GAP = 8
LINE_WEIGHT = 12.0
THREAD_THICKNESS = 1
THREAD_OPACITY = 0.075
BOARD_MARGIN_RATIO = 0.0625
GRAYSCALE_CLIP_LIMIT = 2.6
GRAYSCALE_SHARPEN_SIGMA = 0.85
GRAYSCALE_SHARPEN_AMOUNT = 0.24
DETAIL_EDGE_BOOST = 0.22
DETAIL_EDGE_SIGMA = 1.1
DISPLAY_CONTRAST_LOW = 1.0
DISPLAY_CONTRAST_HIGH = 99.0
DISPLAY_SHARPEN_SIGMA = 0.7
DISPLAY_SHARPEN_AMOUNT = 0.12
PORTRAIT_FACE_FILL = 0.42
PORTRAIT_CROP_SCALE = 1.95
PORTRAIT_VERTICAL_SHIFT = 0.08

SVG_PAGE_FILL = "#ebe7e0"
SVG_BOARD_FILL = "#f8f6f1"
SVG_BOARD_RING = "#cdc5ba"
SVG_BOARD_HIGHLIGHT = "#ffffff"
SVG_THREAD_STROKE = "#181818"
SVG_NAIL_FILL = "#585858"

DEFAULT_PRESET = "portrait"
MAX_HISTORY_ENTRIES = 18
THREAD_WASTE_FACTOR = 1.08
BUILD_MINUTES_PER_LINE = 0.16
BUILD_SETUP_MINUTES = 24
DUO_LAYER_COLORS = ("#7b3f1d", "#1f3a35")
TRIO_LAYER_COLORS = ("#7b3f1d", "#34596c", "#735045")
WORKSPACE_TIPS = {
    "portrait": {
        "title": "Portrait",
        "subtitle": "Balanced detail for people and face-centered artwork.",
        "requested_lines": 3200,
        "default_board_cm": 45,
    },
    "pet": {
        "title": "Pet",
        "subtitle": "A little denser to hold fur and edge detail cleanly.",
        "requested_lines": 3600,
        "default_board_cm": 50,
    },
    "logo": {
        "title": "Logo",
        "subtitle": "Crisp structure for bold shapes, icons, and brand marks.",
        "requested_lines": 2600,
        "default_board_cm": 40,
    },
    "minimal": {
        "title": "Minimal",
        "subtitle": "Cleaner, lighter output for a more open string pattern.",
        "requested_lines": 1800,
        "default_board_cm": 35,
    },
}

app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["MAX_FORM_MEMORY_SIZE"] = MAX_UPLOAD_BYTES
app.request_class.max_form_memory_size = MAX_UPLOAD_BYTES

FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

LAST_JOB = {
    "steps": [],
    "available_lines": 0,
    "selected_lines": DEFAULT_SELECTED_LINES,
    "version": None,
    "session_id": None,
    "session_label": None,
    "preset": DEFAULT_PRESET,
    "cumulative_lengths": [],
}


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(_error):
    message = (
        "The edited image was too large to upload. "
        "The crop export has been reduced, so try generating again."
    )

    if request.path == "/preview" or request.is_json:
        return jsonify({"error": message}), 413

    return render_template(
        "workspace.html",
        **build_workspace_context(error=message),
    ), 413


def ensure_directories():
    Path(app.config["OUTPUT_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(SESSIONS_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    if not Path(HISTORY_FILE).exists():
        Path(HISTORY_FILE).write_text("[]", encoding="utf-8")


def normalize_preset_key(preset_key):
    return preset_key if preset_key in WORKSPACE_TIPS else DEFAULT_PRESET


def get_preset_config(preset_key):
    return WORKSPACE_TIPS[normalize_preset_key(preset_key)]


def serialize_steps(steps):
    return [[int(start), int(end)] for start, end in steps]


def deserialize_steps(serialized_steps):
    return [(int(start), int(end)) for start, end in serialized_steps]


def read_history():
    ensure_directories()
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []

    return payload if isinstance(payload, list) else []


def write_history(entries):
    ensure_directories()
    with open(HISTORY_FILE, "w", encoding="utf-8") as handle:
        json.dump(entries[:MAX_HISTORY_ENTRIES], handle, indent=2)


def create_session_id():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{timestamp}-{uuid4().hex[:6]}"


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


def scaled_board_margin(size):
    return max(2, int(round(size * BOARD_MARGIN_RATIO)))


def create_circle_mask(size):
    margin = scaled_board_margin(size)
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
        return decode_prepared_image(prepared_image_data), True

    if not file_storage or not file_storage.filename:
        raise ValueError("Please upload an image file.")

    return decode_uploaded_file(file_storage), False


def prepare_source_image(image, preserve_view=False):
    if not preserve_view:
        image = auto_focus_portrait(image)
    return center_crop_square(image)


def enhance_reference_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=GRAYSCALE_CLIP_LIMIT, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    soft = cv2.GaussianBlur(gray, (0, 0), GRAYSCALE_SHARPEN_SIGMA)
    gray = cv2.addWeighted(
        gray,
        1.0 + GRAYSCALE_SHARPEN_AMOUNT,
        soft,
        -GRAYSCALE_SHARPEN_AMOUNT,
        0,
    )
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray


def build_display_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(
        gray,
        [DISPLAY_CONTRAST_LOW, DISPLAY_CONTRAST_HIGH],
    )

    if high - low > 1:
        gray = np.clip(
            (gray.astype(np.float32) - low) * (255.0 / (high - low)),
            0,
            255,
        ).astype(np.uint8)

    soft = cv2.GaussianBlur(gray, (0, 0), DISPLAY_SHARPEN_SIGMA)
    gray = cv2.addWeighted(
        gray,
        1.0 + DISPLAY_SHARPEN_AMOUNT,
        soft,
        -DISPLAY_SHARPEN_AMOUNT,
        0,
    )
    return gray


def build_compute_target(gray_image):
    compute_gray = resize_reference(gray_image, COMPUTE_SIZE)
    detail_source = cv2.GaussianBlur(compute_gray, (0, 0), DETAIL_EDGE_SIGMA)
    detail_edges = cv2.Laplacian(detail_source, cv2.CV_32F, ksize=3)
    detail_edges = cv2.convertScaleAbs(np.abs(detail_edges))
    detail_target = cv2.addWeighted(
        compute_gray,
        1.0,
        detail_edges,
        -DETAIL_EDGE_BOOST,
        0,
    )
    detail_target = np.clip(detail_target, 0, 255).astype(np.uint8)
    return mask_reference(detail_target)


def resize_reference(gray_image, size):
    interpolation = (
        cv2.INTER_AREA
        if gray_image.shape[0] >= size
        else cv2.INTER_CUBIC
    )
    return cv2.resize(gray_image, (size, size), interpolation=interpolation)


def mask_reference(gray_image):
    circle_mask = create_circle_mask(gray_image.shape[0])
    return np.where(circle_mask, gray_image, 255).astype(np.uint8)


def preprocess_image(image, preserve_view=False):
    square_image = prepare_source_image(image, preserve_view=preserve_view)
    display_gray = build_display_grayscale(square_image)
    enhanced_gray = enhance_reference_grayscale(square_image)
    gray_compute = build_compute_target(enhanced_gray)
    gray_preview = mask_reference(resize_reference(display_gray, OUTPUT_SIZE))
    return gray_compute, gray_preview


@lru_cache(maxsize=4)
def get_generation_assets(compute_size, output_size, nail_count):
    nails = generate_circle_nails(
        compute_size,
        nail_count,
        margin_ratio=BOARD_MARGIN_RATIO,
    )
    big_nails = generate_circle_nails(
        output_size,
        nail_count,
        margin_ratio=BOARD_MARGIN_RATIO,
    )
    line_cache = build_line_cache(nails, compute_size)
    return nails, big_nails, line_cache


def build_cumulative_lengths(steps):
    if not steps:
        return []

    _, big_nails, _ = get_generation_assets(
        COMPUTE_SIZE,
        OUTPUT_SIZE,
        NAIL_COUNT,
    )
    _, board_radius = get_board_geometry(OUTPUT_SIZE)
    board_diameter = max(board_radius * 2.0, 1.0)

    cumulative_lengths = []
    total_factor = 0.0

    for start_index, end_index in steps:
        start_x, start_y = big_nails[start_index]
        end_x, end_y = big_nails[end_index]
        segment_length = math.dist((start_x, start_y), (end_x, end_y))
        total_factor += (segment_length / board_diameter) * THREAD_WASTE_FACTOR
        cumulative_lengths.append(round(total_factor, 4))

    return cumulative_lengths


def get_thread_factor_for_line_count(selected_lines, cumulative_lengths):
    if not cumulative_lengths or selected_lines <= 0:
        return 0.0

    index = min(selected_lines, len(cumulative_lengths)) - 1
    return float(cumulative_lengths[index])


def estimate_build_minutes(selected_lines):
    return int(round(BUILD_SETUP_MINUTES + selected_lines * BUILD_MINUTES_PER_LINE))


def estimate_materials(selected_lines, cumulative_lengths, board_diameter_cm):
    thread_factor = get_thread_factor_for_line_count(selected_lines, cumulative_lengths)
    thread_length_cm = thread_factor * board_diameter_cm
    build_minutes = estimate_build_minutes(selected_lines)
    return {
        "thread_factor": thread_factor,
        "thread_length_cm": round(thread_length_cm, 1),
        "thread_length_m": round(thread_length_cm / 100.0, 2),
        "build_minutes": build_minutes,
        "build_hours": round(build_minutes / 60.0, 1),
        "board_diameter_cm": board_diameter_cm,
        "nail_count": NAIL_COUNT,
    }


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


def get_board_geometry(size):
    center = (size - 1) / 2.0
    margin = scaled_board_margin(size)
    radius = center - margin
    return center, radius


def build_circle_bezier_segments(center_x, center_y, radius):
    control = radius * 0.552284749831
    return [
        ("move", center_x + radius, center_y),
        (
            "curve",
            center_x + radius,
            center_y + control,
            center_x + control,
            center_y + radius,
            center_x,
            center_y + radius,
        ),
        (
            "curve",
            center_x - control,
            center_y + radius,
            center_x - radius,
            center_y + control,
            center_x - radius,
            center_y,
        ),
        (
            "curve",
            center_x - radius,
            center_y - control,
            center_x - control,
            center_y - radius,
            center_x,
            center_y - radius,
        ),
        (
            "curve",
            center_x + control,
            center_y - radius,
            center_x + radius,
            center_y - control,
            center_x + radius,
            center_y,
        ),
    ]


def format_float(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def build_svg_circle_path(center_x, center_y, radius):
    commands = []
    for segment in build_circle_bezier_segments(center_x, center_y, radius):
        if segment[0] == "move":
            _, x1, y1 = segment
            commands.append(f"M {format_float(x1)} {format_float(y1)}")
            continue

        _, x1, y1, x2, y2, x3, y3 = segment
        commands.append(
            "C "
            f"{format_float(x1)} {format_float(y1)} "
            f"{format_float(x2)} {format_float(y2)} "
            f"{format_float(x3)} {format_float(y3)}"
        )

    commands.append("Z")
    return " ".join(commands)


def build_pdf_circle_path(center_x, center_y, radius):
    commands = []
    for segment in build_circle_bezier_segments(center_x, center_y, radius):
        if segment[0] == "move":
            _, x1, y1 = segment
            commands.append(f"{format_float(x1)} {format_float(y1)} m")
            continue

        _, x1, y1, x2, y2, x3, y3 = segment
        commands.append(
            f"{format_float(x1)} {format_float(y1)} "
            f"{format_float(x2)} {format_float(y2)} "
            f"{format_float(x3)} {format_float(y3)} c"
        )

    commands.append("h")
    return commands


def render_line_svg(steps, line_count):
    _, big_nails, _ = get_generation_assets(
        COMPUTE_SIZE,
        OUTPUT_SIZE,
        NAIL_COUNT,
    )

    center, board_radius = get_board_geometry(OUTPUT_SIZE)
    rim_radius = board_radius + max(12.0, OUTPUT_SIZE * 0.007)
    thread_clip_radius = board_radius - max(12.0, OUTPUT_SIZE * 0.015)
    nail_radius = max(2.8, round(OUTPUT_SIZE / 760.0, 2))
    stroke_width = max(
        0.95,
        round(((THREAD_THICKNESS * OUTPUT_SIZE) / COMPUTE_SIZE) * 0.72, 2),
    )
    clip_path = build_svg_circle_path(center, center, thread_clip_radius)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {OUTPUT_SIZE} {OUTPUT_SIZE}" '
        f'width="{OUTPUT_SIZE}" height="{OUTPUT_SIZE}">'
    ]
    lines.append("<defs>")
    lines.append(
        '<radialGradient id="board-fill" cx="50%" cy="40%" r="60%">'
        '<stop offset="0%" stop-color="#ffffff" />'
        f'<stop offset="100%" stop-color="{SVG_BOARD_FILL}" />'
        "</radialGradient>"
    )
    lines.append(
        '<filter id="board-shadow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="18" stdDeviation="22" flood-color="#000000" flood-opacity="0.18" />'
        "</filter>"
    )
    lines.append(f'<clipPath id="thread-clip"><path d="{clip_path}" /></clipPath>')
    lines.append("</defs>")
    lines.append(
        f'<rect width="{OUTPUT_SIZE}" height="{OUTPUT_SIZE}" fill="{SVG_PAGE_FILL}" />'
    )
    lines.append('<g filter="url(#board-shadow)">')
    lines.append(
        f'<circle cx="{format_float(center)}" cy="{format_float(center)}" '
        f'r="{format_float(rim_radius)}" fill="#efeae2" />'
    )
    lines.append(
        f'<circle cx="{format_float(center)}" cy="{format_float(center)}" '
        f'r="{format_float(board_radius)}" fill="url(#board-fill)" '
        f'stroke="{SVG_BOARD_RING}" stroke-width="10" />'
    )
    lines.append(
        f'<circle cx="{format_float(center)}" cy="{format_float(center)}" '
        f'r="{format_float(board_radius - 18)}" fill="none" '
        f'stroke="{SVG_BOARD_HIGHLIGHT}" stroke-width="20" opacity="0.92" />'
    )
    lines.append("</g>")
    lines.append('<g clip-path="url(#thread-clip)">')
    lines.append(
        f'<circle cx="{format_float(center)}" cy="{format_float(center)}" '
        f'r="{format_float(thread_clip_radius)}" fill="{SVG_BOARD_FILL}" />'
    )
    lines.append(
        f'<g fill="none" stroke="{SVG_THREAD_STROKE}" '
        f'stroke-opacity="{THREAD_OPACITY}" '
        f'stroke-width="{stroke_width}" '
        'shape-rendering="geometricPrecision" '
        'stroke-linecap="round" stroke-linejoin="round">'
    )

    for start_index, end_index in steps[:line_count]:
        start_x, start_y = big_nails[start_index]
        end_x, end_y = big_nails[end_index]
        lines.append(
            f'<line x1="{start_x}" y1="{start_y}" '
            f'x2="{end_x}" y2="{end_y}" />'
        )

    lines.append("</g></g>")
    lines.append(
        f'<g fill="{SVG_NAIL_FILL}" fill-opacity="0.82" '
        'stroke="#f7f4ee" stroke-width="1.3">'
    )
    for nail_x, nail_y in big_nails:
        lines.append(
            f'<circle cx="{nail_x}" cy="{nail_y}" r="{nail_radius}" />'
        )
    lines.append("</g></svg>")
    return "\n".join(lines)


def render_layered_svg(steps, line_count, colors):
    _, big_nails, _ = get_generation_assets(
        COMPUTE_SIZE,
        OUTPUT_SIZE,
        NAIL_COUNT,
    )

    center, board_radius = get_board_geometry(OUTPUT_SIZE)
    rim_radius = board_radius + max(12.0, OUTPUT_SIZE * 0.007)
    thread_clip_radius = board_radius - max(12.0, OUTPUT_SIZE * 0.015)
    nail_radius = max(2.8, round(OUTPUT_SIZE / 760.0, 2))
    stroke_width = max(
        0.95,
        round(((THREAD_THICKNESS * OUTPUT_SIZE) / COMPUTE_SIZE) * 0.72, 2),
    )
    clip_path = build_svg_circle_path(center, center, thread_clip_radius)
    layer_count = max(len(colors), 1)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {OUTPUT_SIZE} {OUTPUT_SIZE}" '
        f'width="{OUTPUT_SIZE}" height="{OUTPUT_SIZE}">'
    ]
    lines.append("<defs>")
    lines.append(
        '<radialGradient id="board-fill" cx="50%" cy="40%" r="60%">'
        '<stop offset="0%" stop-color="#ffffff" />'
        f'<stop offset="100%" stop-color="{SVG_BOARD_FILL}" />'
        "</radialGradient>"
    )
    lines.append(
        '<filter id="board-shadow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="18" stdDeviation="22" flood-color="#000000" flood-opacity="0.18" />'
        "</filter>"
    )
    lines.append(f'<clipPath id="thread-clip"><path d="{clip_path}" /></clipPath>')
    lines.append("</defs>")
    lines.append(
        f'<rect width="{OUTPUT_SIZE}" height="{OUTPUT_SIZE}" fill="{SVG_PAGE_FILL}" />'
    )
    lines.append('<g filter="url(#board-shadow)">')
    lines.append(
        f'<circle cx="{format_float(center)}" cy="{format_float(center)}" '
        f'r="{format_float(rim_radius)}" fill="#efeae2" />'
    )
    lines.append(
        f'<circle cx="{format_float(center)}" cy="{format_float(center)}" '
        f'r="{format_float(board_radius)}" fill="url(#board-fill)" '
        f'stroke="{SVG_BOARD_RING}" stroke-width="10" />'
    )
    lines.append(
        f'<circle cx="{format_float(center)}" cy="{format_float(center)}" '
        f'r="{format_float(board_radius - 18)}" fill="none" '
        f'stroke="{SVG_BOARD_HIGHLIGHT}" stroke-width="20" opacity="0.92" />'
    )
    lines.append("</g>")
    lines.append('<g clip-path="url(#thread-clip)">')
    lines.append(
        f'<circle cx="{format_float(center)}" cy="{format_float(center)}" '
        f'r="{format_float(thread_clip_radius)}" fill="{SVG_BOARD_FILL}" />'
    )

    for layer_index, color in enumerate(colors):
        lines.append(
            f'<g fill="none" stroke="{color}" stroke-opacity="0.55" '
            f'stroke-width="{stroke_width}" shape-rendering="geometricPrecision" '
            'stroke-linecap="round" stroke-linejoin="round">'
        )
        for index, (start_index, end_index) in enumerate(steps[:line_count]):
            if index % layer_count != layer_index:
                continue

            start_x, start_y = big_nails[start_index]
            end_x, end_y = big_nails[end_index]
            lines.append(
                f'<line x1="{start_x}" y1="{start_y}" '
                f'x2="{end_x}" y2="{end_y}" />'
            )
        lines.append("</g>")

    lines.append("</g>")
    lines.append(
        f'<g fill="{SVG_NAIL_FILL}" fill-opacity="0.82" '
        'stroke="#f7f4ee" stroke-width="1.3">'
    )
    for nail_x, nail_y in big_nails:
        lines.append(
            f'<circle cx="{nail_x}" cy="{nail_y}" r="{nail_radius}" />'
        )
    lines.append("</g></svg>")
    return "\n".join(lines)


def build_pdf_document(content_stream):
    stream_bytes = content_stream.encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PDF_OUTPUT_SIZE} {PDF_OUTPUT_SIZE}] "
            "/Resources << /ExtGState << /GS1 5 0 R >> >> "
            "/Contents 4 0 R >>"
        ).encode("ascii"),
        (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        ),
        f"<< /Type /ExtGState /CA {THREAD_OPACITY} /ca {THREAD_OPACITY} >>".encode(
            "ascii"
        ),
    ]

    parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]

    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in parts))
        parts.append(f"{index} 0 obj\n".encode("ascii"))
        parts.append(obj)
        if not obj.endswith(b"\n"):
            parts.append(b"\n")
        parts.append(b"endobj\n")

    start_xref = sum(len(part) for part in parts)
    parts.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    parts.append(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        parts.append(f"{offset:010d} 00000 n \n".encode("ascii"))

    parts.append(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii")
    )
    parts.append(f"startxref\n{start_xref}\n%%EOF\n".encode("ascii"))
    return b"".join(parts)


def render_line_pdf(steps, line_count):
    _, big_nails, _ = get_generation_assets(
        COMPUTE_SIZE,
        OUTPUT_SIZE,
        NAIL_COUNT,
    )

    center, board_radius = get_board_geometry(OUTPUT_SIZE)
    rim_radius = board_radius + max(12.0, OUTPUT_SIZE * 0.007)
    thread_clip_radius = board_radius - max(12.0, OUTPUT_SIZE * 0.015)
    nail_radius = max(2.8, round(OUTPUT_SIZE / 760.0, 2))
    stroke_width = max(
        0.95,
        round(((THREAD_THICKNESS * OUTPUT_SIZE) / COMPUTE_SIZE) * 0.72, 2),
    )
    page_scale = PDF_OUTPUT_SIZE / float(OUTPUT_SIZE)
    scale_value = format_float(page_scale)
    negative_scale_value = format_float(-page_scale)

    content = [
        "q",
        f"{scale_value} 0 0 {negative_scale_value} 0 {PDF_OUTPUT_SIZE} cm",
        "0.922 g",
        f"0 0 {OUTPUT_SIZE} {OUTPUT_SIZE} re f",
        "0.938 g",
        *build_pdf_circle_path(center, center, rim_radius),
        "f",
        "0.982 g",
        *build_pdf_circle_path(center, center, board_radius),
        "f",
        "0.83 G",
        "10 w",
        *build_pdf_circle_path(center, center, board_radius),
        "S",
        "1 G",
        "18 w",
        *build_pdf_circle_path(center, center, board_radius - 18),
        "S",
        "q",
        *build_pdf_circle_path(center, center, thread_clip_radius),
        "W n",
        "/GS1 gs",
        "0.095 G",
        f"{format_float(stroke_width)} w",
        "1 J",
        "1 j",
    ]

    for start_index, end_index in steps[:line_count]:
        start_x, start_y = big_nails[start_index]
        end_x, end_y = big_nails[end_index]
        content.append(
            f"{format_float(start_x)} {format_float(start_y)} m "
            f"{format_float(end_x)} {format_float(end_y)} l S"
        )

    content.extend(["Q", "0.34 g"])
    for nail_x, nail_y in big_nails:
        content.extend(build_pdf_circle_path(nail_x, nail_y, nail_radius))
        content.append("f")

    content.append("Q")
    return build_pdf_document("\n".join(content))


def write_preview_output(steps, selected_lines):
    line_output_path = os.path.join(app.config["OUTPUT_FOLDER"], "line.svg")
    with open(line_output_path, "w", encoding="utf-8") as handle:
        handle.write(render_line_svg(steps, selected_lines))

    line_pdf_output_path = os.path.join(app.config["OUTPUT_FOLDER"], "line.pdf")
    with open(line_pdf_output_path, "wb") as handle:
        handle.write(render_line_pdf(steps, selected_lines))

    duo_output_path = os.path.join(app.config["OUTPUT_FOLDER"], "line_layers_duo.svg")
    with open(duo_output_path, "w", encoding="utf-8") as handle:
        handle.write(render_layered_svg(steps, selected_lines, DUO_LAYER_COLORS))

    trio_output_path = os.path.join(app.config["OUTPUT_FOLDER"], "line_layers_trio.svg")
    with open(trio_output_path, "w", encoding="utf-8") as handle:
        handle.write(render_layered_svg(steps, selected_lines, TRIO_LAYER_COLORS))

    write_steps_file(steps, selected_lines)
    return int(time() * 1000)


def copy_current_outputs_to_session(session_id):
    session_folder = Path(SESSIONS_FOLDER) / session_id
    session_folder.mkdir(parents=True, exist_ok=True)

    file_names = [
        "gray.png",
        "line.svg",
        "line.pdf",
        "line_layers_duo.svg",
        "line_layers_trio.svg",
        "steps.txt",
    ]

    for file_name in file_names:
        source_path = Path(app.config["OUTPUT_FOLDER"]) / file_name
        if source_path.exists():
            shutil.copy2(source_path, session_folder / file_name)

    return {
        "reference": f"sessions/{session_id}/gray.png",
        "result": f"sessions/{session_id}/line.svg",
        "pdf": f"sessions/{session_id}/line.pdf",
        "duo": f"sessions/{session_id}/line_layers_duo.svg",
        "trio": f"sessions/{session_id}/line_layers_trio.svg",
        "steps": f"sessions/{session_id}/steps.txt",
    }


def set_last_job_state(
    steps,
    available_lines,
    selected_lines,
    version,
    *,
    session_id=None,
    session_label=None,
    preset=DEFAULT_PRESET,
    cumulative_lengths=None,
):
    LAST_JOB["steps"] = steps
    LAST_JOB["available_lines"] = available_lines
    LAST_JOB["selected_lines"] = selected_lines
    LAST_JOB["version"] = version
    LAST_JOB["session_id"] = session_id
    LAST_JOB["session_label"] = session_label
    LAST_JOB["preset"] = normalize_preset_key(preset)
    LAST_JOB["cumulative_lengths"] = cumulative_lengths or build_cumulative_lengths(steps)


def save_session_record(
    source_label,
    preset_key,
    requested_lines,
    available_lines,
    selected_lines,
    steps,
    cumulative_lengths,
):
    session_id = create_session_id()
    preset_key = normalize_preset_key(preset_key)
    preset_config = get_preset_config(preset_key)
    asset_paths = copy_current_outputs_to_session(session_id)
    history = read_history()

    entry = {
        "id": session_id,
        "label": source_label or "Untitled Session",
        "preset": preset_key,
        "preset_title": preset_config["title"],
        "requested_lines": int(requested_lines),
        "selected_lines": int(selected_lines),
        "available_lines": int(available_lines),
        "default_board_cm": int(preset_config["default_board_cm"]),
        "created_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "created_at_iso": datetime.now().isoformat(timespec="seconds"),
        "paths": asset_paths,
        "steps": serialize_steps(steps),
        "cumulative_lengths": cumulative_lengths,
    }

    history = [item for item in history if item.get("id") != session_id]
    history.insert(0, entry)
    write_history(history)
    return entry


def update_history_session_selection(session_id, selected_lines):
    history = read_history()
    updated_entry = None

    for entry in history:
        if entry.get("id") != session_id:
            continue

        entry["selected_lines"] = int(selected_lines)
        updated_entry = entry
        break

    if updated_entry:
        write_history(history)

    return updated_entry


def find_history_entry(session_id):
    for entry in read_history():
        if entry.get("id") == session_id:
            return entry
    return None


def activate_history_session(session_id):
    entry = find_history_entry(session_id)
    if not entry:
        return None

    steps = deserialize_steps(entry.get("steps", []))
    if not steps:
        return None

    available_lines = len(steps)
    selected_lines = clamp_line_count(
        entry.get("selected_lines", DEFAULT_SELECTED_LINES),
        available_lines,
    )
    version = write_preview_output(steps, selected_lines)
    cumulative_lengths = entry.get("cumulative_lengths") or build_cumulative_lengths(steps)
    cumulative_lengths = [float(value) for value in cumulative_lengths]

    set_last_job_state(
        steps,
        available_lines,
        selected_lines,
        version,
        session_id=entry.get("id"),
        session_label=entry.get("label"),
        preset=entry.get("preset", DEFAULT_PRESET),
        cumulative_lengths=cumulative_lengths,
    )
    return entry


def build_history_gallery(active_session_id=None):
    gallery = []
    for entry in read_history():
        gallery.append(
            {
                **entry,
                "is_active": entry.get("id") == active_session_id,
            }
        )
    return gallery


def build_workspace_context(
    *,
    error=None,
    requested_lines=None,
    preset_key=DEFAULT_PRESET,
    project_title="",
):
    preset_key = normalize_preset_key(preset_key)
    preset_config = get_preset_config(preset_key)
    initial_lines = (
        clamp_line_count(requested_lines, MAX_LINES)
        if requested_lines is not None
        else preset_config["requested_lines"]
    )
    return {
        "page_id": "workspace",
        "error": error,
        "max_lines": MAX_LINES,
        "requested_lines": initial_lines,
        "presets": WORKSPACE_TIPS,
        "active_preset": preset_key,
        "project_title": project_title,
        "console_available": bool(LAST_JOB["steps"]),
        "app_state": {
            "page": "workspace",
            "preparedExportSize": 2200,
            "preparedExportQuality": 0.94,
            "presetDefaults": {
                key: {
                    "requestedLines": value["requested_lines"],
                    "defaultBoardCm": value["default_board_cm"],
                    "subtitle": value["subtitle"],
                    "title": value["title"],
                }
                for key, value in WORKSPACE_TIPS.items()
            },
        },
    }


def build_active_job_payload():
    if not LAST_JOB["steps"] or not LAST_JOB["available_lines"]:
        return None

    preset_config = get_preset_config(LAST_JOB["preset"])
    history_entry = (
        find_history_entry(LAST_JOB["session_id"])
        if LAST_JOB["session_id"]
        else None
    )
    material_estimate = estimate_materials(
        LAST_JOB["selected_lines"],
        LAST_JOB["cumulative_lengths"],
        preset_config["default_board_cm"],
    )

    return {
        "session_id": LAST_JOB["session_id"],
        "session_label": LAST_JOB["session_label"] or "Current Session",
        "preset": LAST_JOB["preset"],
        "preset_title": preset_config["title"],
        "created_at": history_entry.get("created_at") if history_entry else None,
        "selected_lines": LAST_JOB["selected_lines"],
        "available_lines": LAST_JOB["available_lines"],
        "version": LAST_JOB["version"],
        "reference": "outputs/gray.png",
        "result": "outputs/line.svg",
        "pdf": "outputs/line.pdf",
        "steps": "outputs/steps.txt",
        "duo": "outputs/line_layers_duo.svg",
        "trio": "outputs/line_layers_trio.svg",
        "material_estimate": material_estimate,
    }


def build_console_context(*, empty_message=None):
    active_job = build_active_job_payload()
    active_session_id = active_job["session_id"] if active_job else None
    history_gallery = build_history_gallery(active_session_id=active_session_id)
    default_board_cm = (
        active_job["material_estimate"]["board_diameter_cm"]
        if active_job
        else WORKSPACE_TIPS[DEFAULT_PRESET]["default_board_cm"]
    )

    return {
        "page_id": "console",
        "active_job": active_job,
        "history_gallery": history_gallery,
        "empty_message": empty_message,
        "console_available": bool(active_job),
        "app_state": {
            "page": "console",
            "hasOutput": bool(active_job),
            "previewUrl": url_for("preview"),
            "selectedLines": active_job["selected_lines"] if active_job else 0,
            "availableLines": active_job["available_lines"] if active_job else 0,
            "threadLengthFactors": LAST_JOB["cumulative_lengths"] if active_job else [],
            "defaultBoardCm": default_board_cm,
            "nailCount": NAIL_COUNT,
        },
    }


def generate_string_art(
    image,
    selected_lines,
    *,
    preserve_view=False,
    preset_key=DEFAULT_PRESET,
    source_label="Untitled Session",
):
    ensure_directories()
    preset_key = normalize_preset_key(preset_key)
    requested_line_value = selected_lines

    gray, gray_preview = preprocess_image(image, preserve_view=preserve_view)
    gray_output_path = os.path.join(app.config["OUTPUT_FOLDER"], "gray.png")
    cv2.imwrite(gray_output_path, gray_preview)

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
            LINE_WEIGHT,
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
    cumulative_lengths = build_cumulative_lengths(steps)
    session_entry = save_session_record(
        source_label,
        preset_key,
        clamp_line_count(requested_line_value, MAX_LINES),
        available_lines,
        selected_lines,
        steps,
        cumulative_lengths,
    )
    set_last_job_state(
        steps,
        available_lines,
        selected_lines,
        version,
        session_id=session_entry["id"],
        session_label=session_entry["label"],
        preset=preset_key,
        cumulative_lengths=cumulative_lengths,
    )

    return {
        "session_id": session_entry["id"],
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
    if LAST_JOB["session_id"]:
        update_history_session_selection(LAST_JOB["session_id"], selected_lines)

    return jsonify(
        {
            "image_url": url_for("static", filename="outputs/line.svg", v=version),
            "duo_image_url": url_for(
                "static",
                filename="outputs/line_layers_duo.svg",
                v=version,
            ),
            "trio_image_url": url_for(
                "static",
                filename="outputs/line_layers_trio.svg",
                v=version,
            ),
            "pdf_url": url_for("static", filename="outputs/line.pdf", v=version),
            "steps_url": url_for("static", filename="outputs/steps.txt", v=version),
            "selected_lines": selected_lines,
            "available_lines": LAST_JOB["available_lines"],
            "thread_factor": get_thread_factor_for_line_count(
                selected_lines,
                LAST_JOB["cumulative_lengths"],
            ),
            "build_minutes": estimate_build_minutes(selected_lines),
            "version": version,
        }
    )


@app.route("/")
def home():
    return redirect(url_for("workspace"))


@app.route("/workspace", methods=["GET", "POST"])
def workspace():
    ensure_directories()

    if request.method == "POST":
        file = request.files.get("image")
        prepared_image_data = request.form.get("prepared_image", "")
        preset_key = normalize_preset_key(request.form.get("preset", DEFAULT_PRESET))
        project_title = (request.form.get("project_title") or "").strip()
        requested_lines = request.form.get(
            "requested_lines",
            get_preset_config(preset_key)["requested_lines"],
        )

        try:
            image, preserve_view = load_input_image(file, prepared_image_data)
            source_label = (
                project_title
                or (
                    Path(file.filename).stem
                    if file and getattr(file, "filename", "")
                    else get_preset_config(preset_key)["title"]
                )
            )
            generate_string_art(
                image,
                requested_lines,
                preserve_view=preserve_view,
                preset_key=preset_key,
                source_label=source_label,
            )
        except ValueError as exc:
            return render_template(
                "workspace.html",
                **build_workspace_context(
                    error=str(exc),
                    requested_lines=clamp_line_count(requested_lines, MAX_LINES),
                    preset_key=preset_key,
                    project_title=project_title,
                ),
            )

        return redirect(url_for("console"))

    return render_template("workspace.html", **build_workspace_context())


@app.route("/console")
def console():
    ensure_directories()
    return render_template("console.html", **build_console_context())


@app.route("/console/session/<session_id>")
def console_session(session_id):
    ensure_directories()
    entry = activate_history_session(session_id)
    if not entry:
        return render_template(
            "console.html",
            **build_console_context(
                empty_message="That saved session could not be loaded. Start a new project or choose another saved session."
            ),
        ), 404

    return render_template("console.html", **build_console_context())


if __name__ == "__main__":
    app.run(debug=True)
