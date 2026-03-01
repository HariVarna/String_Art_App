from flask import Flask, render_template, request
import cv2, os, numpy as np
from engine.optimizer import pick_best_nail

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


# ---------------------------
# Generate circular nails
# ---------------------------
def generate_circle_nails(size, count):
    c = size // 2
    r = c - 5
    nails = []

    for i in range(count):
        angle = 2 * np.pi * i / count
        x = int(c + r * np.cos(angle))
        y = int(c + r * np.sin(angle))
        nails.append((x, y))

    return nails


@app.route("/", methods=["GET","POST"])
def index():

    if request.method == "POST":

        # ===== SETTINGS =====
        SIZE = 220          # compute resolution
        OUTPUT_SIZE = 1600  # final image
        NAILS = 300
        LINES = 4000
        THICKNESS = 1
        # ==================

        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # ---------- Preprocess ----------
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (SIZE,SIZE))

        clahe = cv2.createCLAHE(2.5,(8,8))
        gray = clahe.apply(gray)

        cv2.imwrite(os.path.join(OUTPUT_FOLDER,"gray.png"),gray)

        # ---------- Nails ----------
        nails = generate_circle_nails(SIZE, NAILS)
        big_nails = generate_circle_nails(OUTPUT_SIZE, NAILS)

        # ---------- Optimization ----------
        canvas = np.ones((SIZE,SIZE),dtype=np.uint8)*255
        current = 0
        steps = []

        for i in range(LINES):
            nxt = pick_best_nail(gray, canvas, nails, current)
            cv2.line(canvas, nails[current], nails[nxt], 0, 1)
            steps.append((current,nxt))
            current = nxt

            if i % 500 == 0:
                print(i,"/",LINES)

        # ---------- High Res Render ----------
        big_canvas = np.ones((OUTPUT_SIZE,OUTPUT_SIZE),dtype=np.uint8)*255

        for a,b in steps:
            cv2.line(big_canvas, big_nails[a], big_nails[b], 0, THICKNESS)

        cv2.imwrite(os.path.join(OUTPUT_FOLDER,"line.png"),big_canvas)

        # ---------- Save Steps ----------
        with open("static/outputs/steps.txt","w") as f:
            for a,b in steps:
                f.write(f"{a} -> {b}\n")

        return render_template(
            "index.html",
            output="outputs/gray.png",
            line="outputs/line.png",
            steps="outputs/steps.txt"
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
