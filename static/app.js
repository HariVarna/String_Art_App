(function () {
    const state = window.APP_STATE || {};

    const themeToggle = document.getElementById("theme-toggle");
    const page = document.body.dataset.page;

    function getThemeColor(name, fallback = "") {
        const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
        return value || fallback;
    }

    function syncRangeProgress(input) {
        if (!(input instanceof HTMLInputElement) || input.type !== "range") return;
        const min = Number(input.min || 0);
        const max = Number(input.max || 100);
        const value = Number(input.value || min);
        const percentage = max === min ? 100 : ((value - min) / (max - min)) * 100;
        input.style.setProperty("--progress", `${percentage}%`);
    }

    function syncAllRanges(root = document) {
        root.querySelectorAll('input[type="range"]').forEach((input) => syncRangeProgress(input));
    }

    function initTheme() {
        const saved = localStorage.getItem("theme");
        const theme = saved === "dark" || saved === "light" ? saved : "light";
        document.documentElement.setAttribute("theme", theme);

        if (!themeToggle) return;
        themeToggle.addEventListener("click", () => {
            const nextTheme = document.documentElement.getAttribute("theme") === "light" ? "dark" : "light";
            document.documentElement.setAttribute("theme", nextTheme);
            localStorage.setItem("theme", nextTheme);
            if (page === "workspace") {
                const cropCanvas = document.getElementById("crop-canvas");
                if (cropCanvas && cropCanvas._redrawCropCanvas) {
                    cropCanvas._redrawCropCanvas();
                }
            }
        });
    }

    function initRevealAnimations() {
        const revealItems = document.querySelectorAll(".reveal");
        if (!revealItems.length) return;
        document.documentElement.classList.add("js-ready");

        if (!("IntersectionObserver" in window)) {
            revealItems.forEach((item) => item.classList.add("is-visible"));
            return;
        }

        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) return;
                    entry.target.classList.add("is-visible");
                    observer.unobserve(entry.target);
                });
            },
            { threshold: 0.18 }
        );

        revealItems.forEach((item) => observer.observe(item));
    }

    function initViewer() {
        const viewer = document.getElementById("image-viewer");
        const viewerBackdrop = viewer ? viewer.querySelector("[data-close-viewer]") : null;
        const viewerTitle = document.getElementById("image-viewer-title");
        const viewerStage = document.getElementById("image-viewer-stage");
        const viewerImage = document.getElementById("image-viewer-image");
        const viewerZoom = document.getElementById("image-viewer-zoom");
        const viewerZoomValue = document.getElementById("image-viewer-zoom-value");
        const viewerReset = document.getElementById("image-viewer-reset");
        const viewerClose = document.getElementById("image-viewer-close");
        const zoomableItems = document.querySelectorAll(".zoomable-media");

        if (!viewer || !viewerStage || !viewerImage || !zoomableItems.length) return;

        const viewerState = {
            open: false,
            scale: 1,
            x: 0,
            y: 0,
            dragging: false,
            pointerId: null,
            dragStartX: 0,
            dragStartY: 0,
            originX: 0,
            originY: 0,
        };

        function clamp(value, min, max) {
            return Math.min(Math.max(value, min), max);
        }

        function getPanLimits(scale = viewerState.scale) {
            const stageRect = viewerStage.getBoundingClientRect();
            const mediaWidth = viewerImage.naturalWidth || 0;
            const mediaHeight = viewerImage.naturalHeight || 0;
            if (!mediaWidth || !mediaHeight) return { maxX: 0, maxY: 0 };

            const paddedWidth = Math.max(stageRect.width - 32, 1);
            const paddedHeight = Math.max(stageRect.height - 32, 1);
            const fitScale = Math.min(paddedWidth / mediaWidth, paddedHeight / mediaHeight);
            const baseWidth = mediaWidth * fitScale;
            const baseHeight = mediaHeight * fitScale;

            return {
                maxX: Math.max(0, (baseWidth * scale - baseWidth) / 2),
                maxY: Math.max(0, (baseHeight * scale - baseHeight) / 2),
            };
        }

        function applyTransform() {
            const limits = getPanLimits();
            viewerState.x = clamp(viewerState.x, -limits.maxX, limits.maxX);
            viewerState.y = clamp(viewerState.y, -limits.maxY, limits.maxY);

            viewerImage.style.setProperty("--viewer-scale", viewerState.scale.toFixed(3));
            viewerImage.style.setProperty("--viewer-pan-x", `${viewerState.x}px`);
            viewerImage.style.setProperty("--viewer-pan-y", `${viewerState.y}px`);
            viewerStage.classList.toggle("is-draggable", viewerState.scale > 1);
            viewerStage.classList.toggle("is-dragging", viewerState.dragging);

            if (viewerZoom) viewerZoom.value = String(Math.round(viewerState.scale * 100));
            if (viewerZoomValue) viewerZoomValue.textContent = `${Math.round(viewerState.scale * 100)}%`;
        }

        function resetViewer() {
            viewerState.scale = 1;
            viewerState.x = 0;
            viewerState.y = 0;
            applyTransform();
        }

        function openViewer(source) {
            const src = source.currentSrc || source.src;
            if (!src) return;

            viewerState.open = true;
            viewer.hidden = false;
            viewer.setAttribute("aria-hidden", "false");
            document.body.classList.add("viewer-open");
            viewerImage.hidden = false;
            viewerImage.src = src;
            viewerImage.alt = source.alt || "";
            if (viewerTitle) viewerTitle.textContent = source.dataset.zoomLabel || source.alt || "Preview";
            resetViewer();
        }

        function closeViewer() {
            viewerState.open = false;
            viewer.hidden = true;
            viewer.setAttribute("aria-hidden", "true");
            document.body.classList.remove("viewer-open");
            viewerImage.removeAttribute("src");
            viewerImage.hidden = true;
            viewerState.dragging = false;
            viewerState.pointerId = null;
        }

        zoomableItems.forEach((item) => {
            item.addEventListener("click", () => openViewer(item));
            item.addEventListener("keydown", (event) => {
                if (event.key !== "Enter" && event.key !== " ") return;
                event.preventDefault();
                openViewer(item);
            });
        });

        if (viewerBackdrop) viewerBackdrop.addEventListener("click", closeViewer);
        if (viewerClose) viewerClose.addEventListener("click", closeViewer);
        if (viewerReset) viewerReset.addEventListener("click", resetViewer);

        if (viewerZoom) {
            viewerZoom.addEventListener("input", () => {
                viewerState.scale = clamp(Number(viewerZoom.value) / 100, 1, 16);
                applyTransform();
            });
            syncRangeProgress(viewerZoom);
        }

        viewerImage.addEventListener("load", resetViewer);

        viewerStage.addEventListener(
            "wheel",
            (event) => {
                if (!viewerState.open) return;
                event.preventDefault();
                const delta = event.deltaY < 0 ? 0.16 : -0.16;
                viewerState.scale = clamp(viewerState.scale + delta, 1, 16);
                applyTransform();
            },
            { passive: false }
        );

        viewerStage.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 || viewerState.scale <= 1) return;
            viewerState.dragging = true;
            viewerState.pointerId = event.pointerId;
            viewerState.dragStartX = event.clientX;
            viewerState.dragStartY = event.clientY;
            viewerState.originX = viewerState.x;
            viewerState.originY = viewerState.y;
            viewerStage.setPointerCapture(event.pointerId);
            applyTransform();
        });

        viewerStage.addEventListener("pointermove", (event) => {
            if (!viewerState.dragging || viewerState.pointerId !== event.pointerId) return;
            viewerState.x = viewerState.originX + (event.clientX - viewerState.dragStartX);
            viewerState.y = viewerState.originY + (event.clientY - viewerState.dragStartY);
            applyTransform();
        });

        ["pointerup", "pointercancel"].forEach((eventName) => {
            viewerStage.addEventListener(eventName, (event) => {
                if (!viewerState.dragging || viewerState.pointerId !== event.pointerId) return;
                if (viewerStage.hasPointerCapture(event.pointerId)) {
                    viewerStage.releasePointerCapture(event.pointerId);
                }
                viewerState.dragging = false;
                viewerState.pointerId = null;
                applyTransform();
            });
        });

        window.addEventListener("keydown", (event) => {
            if (event.key === "Escape" && viewerState.open) {
                closeViewer();
            }
        });

        window.addEventListener("resize", () => {
            if (viewerState.open) applyTransform();
        });
    }

    function initWorkspace() {
        const form = document.getElementById("art-form");
        const fileInput = document.getElementById("image-input");
        const presetInput = document.getElementById("preset-input");
        const presetButtons = document.querySelectorAll("[data-preset-key]");
        const presetHeadline = document.getElementById("preset-headline");
        const presetSubtitle = document.getElementById("preset-subtitle");
        const presetBoardLabel = document.getElementById("preset-board-label");
        const requestedLines = document.getElementById("requested-lines");
        const requestedLinesValue = document.getElementById("requested-lines-value");
        const zoomRange = document.getElementById("zoom-range");
        const zoomValue = document.getElementById("zoom-value");
        const generateButton = document.getElementById("generate-button");
        const resetCropButton = document.getElementById("reset-crop");
        const preparedImageInput = document.getElementById("prepared-image");
        const uploadQualityBadge = document.getElementById("upload-quality-badge");
        const uploadQualityTip = document.getElementById("upload-quality-tip");
        const uploadQualityMeta = document.getElementById("upload-quality-meta");
        const cropCanvas = document.getElementById("crop-canvas");
        const cropHint = document.getElementById("crop-hint");

        if (!form || !cropCanvas || !(cropCanvas instanceof HTMLCanvasElement)) return;

        const cropContext = cropCanvas.getContext("2d");
        const presetDefaults = state.presetDefaults || {};
        const preparedExportSize = Number(state.preparedExportSize || 2200);
        const preparedExportQuality = Number(state.preparedExportQuality || 0.94);

        const cropState = {
            image: null,
            baseScale: 1,
            scale: 1,
            x: 0,
            y: 0,
            dragging: false,
            pointerId: null,
            dragStartX: 0,
            dragStartY: 0,
            originX: 0,
            originY: 0,
        };

        function setUploadQualityState(quality, label, tip, meta) {
            if (uploadQualityBadge) {
                uploadQualityBadge.dataset.quality = quality;
                uploadQualityBadge.textContent = label;
            }
            if (uploadQualityTip) uploadQualityTip.textContent = tip;
            if (uploadQualityMeta) uploadQualityMeta.textContent = meta;
        }

        function resetUploadQualityState() {
            setUploadQualityState(
                "ready",
                "High quality recommended",
                "Best results usually come from a crisp portrait or logo with clean subject separation.",
                "Recommended: at least 1600 px on the shortest side."
            );
        }

        function formatFileSize(bytes) {
            if (!Number.isFinite(bytes) || bytes <= 0) return "0 KB";
            if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
            return `${Math.max(1, Math.round(bytes / 1024))} KB`;
        }

        function evaluateUploadQuality(image, file) {
            const shortestSide = Math.min(image.width, image.height);
            const meta = `${image.width} x ${image.height} | ${formatFileSize(file ? file.size : 0)}`;

            if (shortestSide >= 1800) {
                return {
                    quality: "high",
                    label: "High detail",
                    tip: "Excellent input. This upload should preserve strong edge detail for a cleaner string pattern.",
                    meta,
                };
            }

            if (shortestSide >= 1200) {
                return {
                    quality: "good",
                    label: "Good detail",
                    tip: "This should work well, though an even sharper image can still improve the result.",
                    meta,
                };
            }

            return {
                quality: "low",
                label: "Low detail",
                tip: "Try a higher-resolution image if possible. Small or blurry uploads usually lose detail in the final path.",
                meta,
            };
        }

        function applyPreset(button) {
            const presetKey = button.dataset.presetKey;
            const presetTitle = button.dataset.presetTitle || "Preset";
            const presetSubtitleText = button.dataset.presetSubtitle || "";
            const requestedLineValue = button.dataset.presetLines || requestedLines.value;
            const boardCm = button.dataset.presetBoard || "45";

            presetButtons.forEach((item) => item.classList.toggle("is-active", item === button));
            if (presetInput) presetInput.value = presetKey;
            if (requestedLines) {
                requestedLines.value = requestedLineValue;
                requestedLinesValue.textContent = requestedLines.value;
                syncRangeProgress(requestedLines);
            }
            if (presetHeadline) presetHeadline.textContent = `${presetTitle} workspace`;
            if (presetSubtitle) presetSubtitle.textContent = presetSubtitleText;
            if (presetBoardLabel) {
                presetBoardLabel.textContent = `${boardCm} cm board diameter is a strong starting point for this preset.`;
            }
        }

        presetButtons.forEach((button) => {
            button.addEventListener("click", () => applyPreset(button));
        });

        function drawCropGrid(width, height, radius) {
            const centerX = width / 2;
            const centerY = height / 2;
            const left = centerX - radius;
            const top = centerY - radius;
            const diameter = radius * 2;

            cropContext.save();
            cropContext.beginPath();
            cropContext.arc(centerX, centerY, radius, 0, Math.PI * 2);
            cropContext.clip();
            cropContext.lineWidth = 1;
            cropContext.setLineDash([6, 6]);
            cropContext.strokeStyle = getThemeColor("--crop-grid", "rgba(255,255,255,0.08)");

            for (let index = 1; index < 3; index += 1) {
                const x = left + (diameter / 3) * index;
                const y = top + (diameter / 3) * index;
                cropContext.beginPath();
                cropContext.moveTo(x, top);
                cropContext.lineTo(x, top + diameter);
                cropContext.stroke();
                cropContext.beginPath();
                cropContext.moveTo(left, y);
                cropContext.lineTo(left + diameter, y);
                cropContext.stroke();
            }

            cropContext.restore();
        }

        function constrainCropImage() {
            if (!cropState.image) return;
            const scaledWidth = cropState.image.width * cropState.scale;
            const scaledHeight = cropState.image.height * cropState.scale;
            if (scaledWidth <= cropCanvas.width) cropState.x = (cropCanvas.width - scaledWidth) / 2;
            else cropState.x = Math.min(0, Math.max(cropState.x, cropCanvas.width - scaledWidth));
            if (scaledHeight <= cropCanvas.height) cropState.y = (cropCanvas.height - scaledHeight) / 2;
            else cropState.y = Math.min(0, Math.max(cropState.y, cropCanvas.height - scaledHeight));
        }

        function drawCropCanvas() {
            const { width, height } = cropCanvas;
            cropContext.clearRect(0, 0, width, height);
            cropContext.fillStyle = getThemeColor("--crop-bg", "#2a1d18");
            cropContext.fillRect(0, 0, width, height);

            if (cropState.image) {
                cropContext.drawImage(
                    cropState.image,
                    cropState.x,
                    cropState.y,
                    cropState.image.width * cropState.scale,
                    cropState.image.height * cropState.scale
                );
            } else {
                cropContext.fillStyle = getThemeColor("--text", "#f4e8d6");
                cropContext.font = '18px "Aptos"';
                cropContext.textAlign = "center";
                cropContext.fillText("Your crop preview appears here", width / 2, height / 2 - 8);
                cropContext.fillStyle = getThemeColor("--muted", "#c8b39a");
                cropContext.font = '14px "Aptos"';
                cropContext.fillText("Upload an image to drag and zoom it", width / 2, height / 2 + 20);
            }

            const radius = width * 0.44;
            drawCropGrid(width, height, radius);
            cropContext.save();
            cropContext.fillStyle = getThemeColor("--crop-overlay", "rgba(18,12,10,0.54)");
            cropContext.beginPath();
            cropContext.rect(0, 0, width, height);
            cropContext.arc(width / 2, height / 2, radius, 0, Math.PI * 2, true);
            cropContext.fill("evenodd");
            cropContext.restore();
            cropContext.strokeStyle = getThemeColor("--crop-ring", "rgba(240,207,154,0.92)");
            cropContext.lineWidth = 2;
            cropContext.beginPath();
            cropContext.arc(width / 2, height / 2, radius, 0, Math.PI * 2);
            cropContext.stroke();
        }

        cropCanvas._redrawCropCanvas = drawCropCanvas;

        function resetCropPosition() {
            if (!cropState.image) {
                drawCropCanvas();
                return;
            }
            cropState.baseScale = Math.max(cropCanvas.width / cropState.image.width, cropCanvas.height / cropState.image.height);
            cropState.scale = cropState.baseScale * (Number(zoomRange.value) / 100);
            cropState.x = (cropCanvas.width - cropState.image.width * cropState.scale) / 2;
            cropState.y = (cropCanvas.height - cropState.image.height * cropState.scale) / 2;
            constrainCropImage();
            drawCropCanvas();
        }

        function exportPreparedImage() {
            if (!cropState.image) return "";
            const exportCanvas = document.createElement("canvas");
            exportCanvas.width = preparedExportSize;
            exportCanvas.height = preparedExportSize;
            const exportContext = exportCanvas.getContext("2d");
            const factor = exportCanvas.width / cropCanvas.width;
            exportContext.fillStyle = "#ffffff";
            exportContext.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
            exportContext.drawImage(
                cropState.image,
                cropState.x * factor,
                cropState.y * factor,
                cropState.image.width * cropState.scale * factor,
                cropState.image.height * cropState.scale * factor
            );
            return exportCanvas.toDataURL("image/jpeg", preparedExportQuality);
        }

        function stopDragging(pointerId) {
            if (!cropState.dragging || cropState.pointerId !== pointerId) return;
            if (cropCanvas.hasPointerCapture(pointerId)) {
                cropCanvas.releasePointerCapture(pointerId);
            }
            cropState.dragging = false;
            cropState.pointerId = null;
            cropCanvas.classList.remove("dragging");
        }

        function loadSelectedImage(file) {
            const objectUrl = URL.createObjectURL(file);
            const image = new Image();
            image.onload = () => {
                cropState.image = image;
                zoomRange.value = "100";
                zoomValue.textContent = "100%";
                syncRangeProgress(zoomRange);
                if (cropHint) cropHint.textContent = "Drag inside the crop area to position the subject.";
                const qualityState = evaluateUploadQuality(image, file);
                setUploadQualityState(qualityState.quality, qualityState.label, qualityState.tip, qualityState.meta);
                resetCropPosition();
                URL.revokeObjectURL(objectUrl);
            };
            image.src = objectUrl;
        }

        if (fileInput) {
            fileInput.addEventListener("change", () => {
                const [file] = fileInput.files || [];
                if (!file) {
                    cropState.image = null;
                    if (cropHint) cropHint.textContent = "Load an image to start positioning it.";
                    resetUploadQualityState();
                    drawCropCanvas();
                    return;
                }
                preparedImageInput.value = "";
                loadSelectedImage(file);
            });
        }

        if (zoomRange) {
            zoomRange.addEventListener("input", () => {
                zoomValue.textContent = `${zoomRange.value}%`;
                syncRangeProgress(zoomRange);
                if (cropState.image) resetCropPosition();
            });
        }

        if (requestedLines) {
            requestedLines.addEventListener("input", () => {
                requestedLinesValue.textContent = requestedLines.value;
                syncRangeProgress(requestedLines);
            });
        }

        if (resetCropButton) {
            resetCropButton.addEventListener("click", () => {
                if (zoomRange) {
                    zoomRange.value = "100";
                    zoomValue.textContent = "100%";
                    syncRangeProgress(zoomRange);
                }
                resetCropPosition();
            });
        }

        cropCanvas.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 || !cropState.image) return;
            cropState.dragging = true;
            cropState.pointerId = event.pointerId;
            cropState.dragStartX = event.clientX;
            cropState.dragStartY = event.clientY;
            cropState.originX = cropState.x;
            cropState.originY = cropState.y;
            cropCanvas.classList.add("dragging");
            cropCanvas.setPointerCapture(event.pointerId);
        });

        cropCanvas.addEventListener("pointermove", (event) => {
            if (!cropState.dragging || cropState.pointerId !== event.pointerId) return;
            cropState.x = cropState.originX + (event.clientX - cropState.dragStartX);
            cropState.y = cropState.originY + (event.clientY - cropState.dragStartY);
            constrainCropImage();
            drawCropCanvas();
        });

        ["pointerup", "pointercancel", "pointerleave"].forEach((eventName) => {
            cropCanvas.addEventListener(eventName, (event) => stopDragging(event.pointerId));
        });

        form.addEventListener("submit", () => {
            if (cropState.image) {
                preparedImageInput.value = exportPreparedImage();
            }
            if (generateButton) {
                generateButton.disabled = true;
                generateButton.textContent = "Generating Artboard";
            }
        });

        const activePresetButton = document.querySelector(".preset-card.is-active") || presetButtons[0];
        if (activePresetButton) applyPreset(activePresetButton);
        resetUploadQualityState();
        drawCropCanvas();
        syncAllRanges(form);
    }

    function initConsole() {
        const previewLines = document.getElementById("preview-lines");
        const previewLinesValue = document.getElementById("preview-lines-value");
        const previewStatus = document.getElementById("preview-status");
        const previewStatusCount = document.getElementById("preview-status-count");
        const availableLinesCount = document.getElementById("available-lines-count");
        const lineImage = document.getElementById("line-image");
        const layerPreviewImage = document.getElementById("layer-preview-image");
        const linePdfLink = document.getElementById("line-pdf-link");
        const stepsLink = document.getElementById("steps-link");
        const duoLink = document.getElementById("duo-link");
        const trioLink = document.getElementById("trio-link");
        const layerButtons = document.querySelectorAll("[data-layer-mode]");
        const boardSizeRange = document.getElementById("board-size-range");
        const boardSizeValue = document.getElementById("board-size-value");
        const estimateThread = document.getElementById("estimate-thread");
        const estimateHours = document.getElementById("estimate-hours");
        const estimateNails = document.getElementById("estimate-nails");
        const estimateLines = document.getElementById("estimate-lines");
        const estimateBoard = document.getElementById("estimate-board");
        const compareTriggers = document.querySelectorAll(".compare-trigger");
        const compareTargetImage = document.getElementById("compare-target-image");
        const compareTargetTitle = document.getElementById("compare-target-title");
        const comparePlaceholder = document.getElementById("compare-placeholder");
        const compareTargetFrame = document.getElementById("compare-target-frame");

        if (!state.hasOutput) {
            syncAllRanges();
            return;
        }

        let previewTimeout = null;
        let previewRequestId = 0;
        let activeLayerMode = "duo";
        const threadLengthFactors = Array.isArray(state.threadLengthFactors) ? state.threadLengthFactors.map(Number) : [];

        function getCurrentThreadFactor(lineCount) {
            if (!threadLengthFactors.length) return 0;
            const index = Math.max(0, Math.min(threadLengthFactors.length - 1, lineCount - 1));
            return Number(threadLengthFactors[index] || 0);
        }

        function updateEstimator() {
            if (!boardSizeRange || !estimateThread || !estimateHours || !estimateLines || !estimateBoard || !estimateNails) return;

            const lineCount = previewLines ? Number(previewLines.value) : Number(state.selectedLines || 0);
            const boardCm = Number(boardSizeRange.value || state.defaultBoardCm || 45);
            const threadFactor = getCurrentThreadFactor(lineCount);
            const threadMeters = (threadFactor * boardCm) / 100;
            const buildHours = Math.round((24 + lineCount * 0.16) / 6) / 10;

            boardSizeValue.textContent = `${boardCm} cm`;
            estimateThread.textContent = `${threadMeters.toFixed(2)} m`;
            estimateHours.textContent = `${buildHours.toFixed(1)} h`;
            estimateNails.textContent = String(state.nailCount || 360);
            estimateLines.textContent = String(lineCount);
            estimateBoard.textContent = `${boardCm} cm`;
            syncRangeProgress(boardSizeRange);
        }

        function applyLayerMode(mode) {
            activeLayerMode = mode;
            layerButtons.forEach((button) => button.classList.toggle("is-active", button.dataset.layerMode === mode));
            if (!layerPreviewImage) return;
            if (mode === "trio") {
                layerPreviewImage.src = layerPreviewImage.dataset.trioSrc || layerPreviewImage.src;
                layerPreviewImage.alt = "Multi-layer string art preview";
                layerPreviewImage.dataset.zoomLabel = "Multi-Layer Preview";
            } else {
                layerPreviewImage.src = layerPreviewImage.dataset.duoSrc || layerPreviewImage.src;
                layerPreviewImage.alt = "Two-color string art preview";
                layerPreviewImage.dataset.zoomLabel = "Two-Color Preview";
            }
        }

        async function updatePreview(lineCount) {
            const currentRequest = ++previewRequestId;

            try {
                const response = await fetch(state.previewUrl || "/preview", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ line_count: lineCount }),
                });
                const data = await response.json();
                if (currentRequest !== previewRequestId) return;
                if (!response.ok) throw new Error(data.error || "Preview failed.");

                if (lineImage) {
                    lineImage.style.opacity = "0";
                    lineImage.onload = () => {
                        lineImage.style.opacity = "1";
                    };
                    lineImage.src = data.image_url;
                }

                if (layerPreviewImage) {
                    layerPreviewImage.dataset.duoSrc = data.duo_image_url;
                    layerPreviewImage.dataset.trioSrc = data.trio_image_url;
                    applyLayerMode(activeLayerMode);
                }

                if (linePdfLink) linePdfLink.href = data.pdf_url;
                if (stepsLink) stepsLink.href = data.steps_url;
                if (duoLink) duoLink.href = data.duo_image_url;
                if (trioLink) trioLink.href = data.trio_image_url;
                if (previewLinesValue) previewLinesValue.textContent = String(data.selected_lines);
                if (previewStatus) previewStatus.textContent = `Showing ${data.selected_lines} of ${data.available_lines} generated lines.`;
                if (previewStatusCount) previewStatusCount.textContent = String(data.selected_lines);
                if (availableLinesCount) availableLinesCount.textContent = String(data.available_lines);
                updateEstimator();
            } catch (error) {
                if (previewStatus) previewStatus.textContent = error.message;
            }
        }

        if (previewLines) {
            previewLines.addEventListener("input", () => {
                previewLinesValue.textContent = previewLines.value;
                if (previewStatus) {
                    previewStatus.textContent = `Rendering ${previewLines.value} lines...`;
                }
                syncRangeProgress(previewLines);
                updateEstimator();
                clearTimeout(previewTimeout);
                previewTimeout = setTimeout(() => updatePreview(Number(previewLines.value)), 120);
            });
        }

        layerButtons.forEach((button) => {
            button.addEventListener("click", () => applyLayerMode(button.dataset.layerMode || "duo"));
        });

        if (boardSizeRange) {
            boardSizeRange.addEventListener("input", updateEstimator);
        }

        compareTriggers.forEach((trigger) => {
            trigger.addEventListener("click", () => {
                if (!compareTargetImage || !compareTargetTitle) return;
                compareTargetImage.src = trigger.dataset.compareImage || "";
                compareTargetImage.dataset.zoomLabel = trigger.dataset.compareTitle || "Saved Session Comparison";
                compareTargetImage.alt = `${trigger.dataset.compareTitle || "Selected Session"} comparison output`;
                compareTargetImage.hidden = false;
                compareTargetTitle.textContent = trigger.dataset.compareTitle || "Selected Session";
                if (comparePlaceholder) comparePlaceholder.hidden = true;
                if (compareTargetFrame) compareTargetFrame.classList.remove("compare-frame-empty");
            });
        });

        applyLayerMode(activeLayerMode);
        updateEstimator();
        syncAllRanges();
    }

    initTheme();
    initRevealAnimations();
    initViewer();
    syncAllRanges();

    if (page === "workspace") initWorkspace();
    if (page === "console") initConsole();
})();
