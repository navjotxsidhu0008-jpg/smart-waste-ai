/**
 * ♻️ AI Smart Waste Management System — Frontend Application Logic (static/js/app.js)
 *
 * Handles:
 * - Real-time webcam initialization & graceful error handling
 * - Non-blocking throttled frame capture (224x224 JPEG) to POST /api/predict
 * - Live updates of waste category, confidence meter, smart bin recommendation & 6-class probabilities
 * - Automatic refresh of SQLite statistics counters & recent detections table
 * - Photo upload and interactive classroom sample simulation
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements — Camera Controls
    const videoEl = document.getElementById("webcam-video");
    const canvasEl = document.getElementById("capture-canvas");
    const uploadedPreviewEl = document.getElementById("uploaded-preview");
    const placeholderEl = document.getElementById("camera-placeholder");
    const scanLaserEl = document.getElementById("scan-laser");
    const reticleHintEl = document.getElementById("reticle-hint");
    const fpsBadgeEl = document.getElementById("fps-badge");

    const startCameraBtn = document.getElementById("start-camera-btn");
    const stopCameraBtn = document.getElementById("stop-camera-btn");
    const captureOnceBtn = document.getElementById("capture-once-btn");
    const placeholderStartBtn = document.getElementById("placeholder-start-btn");
    const heroStartBtn = document.getElementById("hero-start-camera-btn");
    const imageUploadInput = document.getElementById("image-upload-input");
    const intervalSelect = document.getElementById("interval-select");

    const errorBoxEl = document.getElementById("camera-error-box");
    const errorTitleEl = document.getElementById("camera-error-title");
    const errorMsgEl = document.getElementById("camera-error-msg");

    // Navbar Status Elements
    const cameraStatusDot = document.getElementById("camera-status-dot");
    const cameraStatusText = document.getElementById("camera-status-text");

    // Prediction Card Elements
    const predBannerEl = document.getElementById("predicted-banner");
    const predIconEl = document.getElementById("pred-icon");
    const predCategoryEl = document.getElementById("pred-category");
    const predObjectSubEl = document.getElementById("pred-object-sub");
    const predConfidenceEl = document.getElementById("pred-confidence");
    const confBarFillEl = document.getElementById("confidence-bar-fill");
    const lowConfAlertEl = document.getElementById("low-confidence-alert");
    const lowConfTextEl = document.getElementById("low-confidence-text");
    const dbLogBadgeEl = document.getElementById("db-log-badge");

    const recBinIconEl = document.getElementById("rec-bin-icon");
    const recBinNameEl = document.getElementById("rec-bin-name");
    const recInstructionEl = document.getElementById("rec-instruction");
    const recEcoTipEl = document.getElementById("rec-eco-tip");

    // Dashboard & Table Elements
    const refreshStatsBtn = document.getElementById("refresh-stats-btn");
    const clearDbBtn = document.getElementById("clear-db-btn");
    const recentTbodyEl = document.getElementById("recent-detections-tbody");

    // State
    let mediaStream = null;
    let captureTimer = null;
    let isProcessingFrame = false;
    let captureIntervalMs = (window.APP_CONFIG && window.APP_CONFIG.captureIntervalMs) || 1200;

    // ----------------------------------------------------------------------
    // 1. WEBCAM MANAGEMENT
    // ----------------------------------------------------------------------

    async function startCamera() {
        hideCameraError();

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showCameraError(
                "Browser Not Supported",
                "Your browser does not support webcam access (getUserMedia). Please use a modern browser like Chrome, Edge, or Firefox, or test using 'Upload Photo' below."
            );
            return;
        }

        try {
            fpsBadgeEl.textContent = "Requesting camera...";
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "environment"
                },
                audio: false
            });

            videoEl.srcObject = mediaStream;
            await videoEl.play();

            // Update UI for active camera
            uploadedPreviewEl.style.display = "none";
            videoEl.style.display = "block";
            placeholderEl.style.display = "none";
            scanLaserEl.classList.add("active");
            reticleHintEl.textContent = "AI Scanning Active — Hold waste item in center";

            startCameraBtn.disabled = true;
            stopCameraBtn.disabled = false;
            captureOnceBtn.disabled = false;

            cameraStatusDot.className = "dot dot-active";
            cameraStatusText.textContent = "Camera: Live";
            fpsBadgeEl.textContent = `Scanning (${(captureIntervalMs / 1000).toFixed(1)}s)`;
            fpsBadgeEl.className = "badge badge-success";

            // Trigger initial capture after 400ms warm-up, then start controlled loop
            setTimeout(() => {
                if (mediaStream) captureAndPredictFrame(false);
            }, 400);

            startCaptureLoop();
        } catch (err) {
            console.error("Webcam access error:", err);
            cameraStatusDot.className = "dot dot-error";
            cameraStatusText.textContent = "Camera: Blocked";
            fpsBadgeEl.textContent = "Camera Error";
            fpsBadgeEl.className = "badge badge-warning";

            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                showCameraError(
                    "Camera Permission Denied",
                    "Camera permission was denied. Please allow camera access in your browser address bar, or use 'Upload Photo' / 'Workshop Quick Test Samples' below."
                );
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                showCameraError(
                    "No Camera Detected",
                    "No webcam was found on this computer. You can still demonstrate all AI features using 'Upload Photo' or the 'Workshop Quick Test Samples' below."
                );
            } else {
                showCameraError(
                    "Camera Unavailable",
                    `Could not start video source (${err.message || err.name}). Please check if another app is using the webcam.`
                );
            }
        }
    }

    function stopCamera() {
        stopCaptureLoop();

        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => track.stop());
            mediaStream = null;
        }

        videoEl.srcObject = null;
        scanLaserEl.classList.remove("active");
        placeholderEl.style.display = "flex";
        reticleHintEl.textContent = "Center waste item inside frame";

        startCameraBtn.disabled = false;
        stopCameraBtn.disabled = true;
        captureOnceBtn.disabled = true;

        cameraStatusDot.className = "dot dot-idle";
        cameraStatusText.textContent = "Camera: Idle";
        fpsBadgeEl.textContent = "Standby";
        fpsBadgeEl.className = "badge badge-neutral";
    }

    function startCaptureLoop() {
        stopCaptureLoop();
        captureTimer = setInterval(() => {
            if (mediaStream && !isProcessingFrame) {
                captureAndPredictFrame(false);
            }
        }, captureIntervalMs);
    }

    function stopCaptureLoop() {
        if (captureTimer) {
            clearInterval(captureTimer);
            captureTimer = null;
        }
    }

    function showCameraError(title, message) {
        errorTitleEl.textContent = title;
        errorMsgEl.textContent = message;
        errorBoxEl.style.display = "flex";
    }

    function hideCameraError() {
        errorBoxEl.style.display = "none";
    }

    // ----------------------------------------------------------------------
    // 2. FRAME CAPTURE & API PREDICTION
    // ----------------------------------------------------------------------

    function captureAndPredictFrame(forceSave = false, demoHint = null) {
        if (isProcessingFrame) return;
        if (!mediaStream && !demoHint) return;

        const ctx = canvasEl.getContext("2d");
        canvasEl.width = 224;
        canvasEl.height = 224;

        if (mediaStream && videoEl.readyState >= 2) {
            // Center-crop square from video feed for distortion-free 224x224 input
            const vw = videoEl.videoWidth || 640;
            const vh = videoEl.videoHeight || 480;
            const minDim = Math.min(vw, vh);
            const sx = (vw - minDim) / 2;
            const sy = (vh - minDim) / 2;
            ctx.drawImage(videoEl, sx, sy, minDim, minDim, 0, 0, 224, 224);
        } else {
            // Render synthetic visual card on canvas for classroom preset testing when camera is off
            renderPresetSampleCanvas(ctx, demoHint || "Plastic");
            // Also show the rendered canvas in the preview viewport
            uploadedPreviewEl.src = canvasEl.toDataURL("image/jpeg", 0.9);
            uploadedPreviewEl.style.display = "block";
            videoEl.style.display = "none";
            placeholderEl.style.display = "none";
        }

        isProcessingFrame = true;
        canvasEl.toBlob(
            async (blob) => {
                try {
                    if (!blob) return;
                    await sendImageBlobToApi(blob, forceSave, demoHint);
                } catch (err) {
                    console.error("Frame processing error:", err);
                } finally {
                    isProcessingFrame = false;
                }
            },
            "image/jpeg",
            0.85
        );
    }

    async function sendImageBlobToApi(blob, forceSave = false, demoHint = null) {
        const formData = new FormData();
        formData.append("file", blob, "webcam_frame.jpg");
        if (forceSave) {
            formData.append("force_save", "true");
        }
        if (demoHint) {
            formData.append("demo_hint", demoHint);
        }

        try {
            dbLogBadgeEl.textContent = "Analyzing...";
            dbLogBadgeEl.className = "badge badge-info";

            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error (${response.status})`);
            }

            const result = await response.json();
            updatePredictionUI(result);

            if (result.saved_to_db) {
                await refreshDashboardStats();
            }
        } catch (err) {
            console.error("Prediction API error:", err);
            dbLogBadgeEl.textContent = "API Error";
            dbLogBadgeEl.className = "badge badge-warning";
        }
    }

    // ----------------------------------------------------------------------
    // 3. UPDATE PREDICTION & SMART BIN UI
    // ----------------------------------------------------------------------

    function updatePredictionUI(data) {
        // Main Banner
        predIconEl.textContent = data.icon || "♻️";
        predCategoryEl.textContent = data.waste_type;
        predBannerEl.style.setProperty("--active-color", data.bin_color || "#10b981");

        if (data.detected_object) {
            predObjectSubEl.textContent = `Visual Feature: ${data.detected_object}`;
        } else {
            predObjectSubEl.textContent = data.status_message || "Classification complete";
        }

        // Confidence Percentage & Progress Bar
        const conf = Number(data.confidence || 0).toFixed(1);
        predConfidenceEl.textContent = `${conf}%`;
        predConfidenceEl.style.color = data.is_low_confidence ? "#d97706" : (data.bin_color || "#059669");
        confBarFillEl.style.width = `${Math.min(100, Math.max(0, conf))}%`;
        confBarFillEl.style.background = data.is_low_confidence
            ? "#f59e0b"
            : `linear-gradient(90deg, ${data.bin_color || "#10b981"}, #06b6d4)`;

        // Low-Confidence Alert
        if (data.is_low_confidence) {
            lowConfTextEl.textContent = data.status_message || "Uncertain prediction — please show the object more clearly.";
            lowConfAlertEl.style.display = "flex";
        } else {
            lowConfAlertEl.style.display = "none";
        }

        // Smart Bin Recommendation
        recBinIconEl.textContent = data.icon || "🗑️";
        recBinNameEl.textContent = data.recommended_bin;
        recBinNameEl.style.color = data.bin_color || "#0f172a";
        recInstructionEl.textContent = data.instruction;
        recEcoTipEl.textContent = data.eco_tip;

        // Database Log Status Pill
        if (data.saved_to_db) {
            dbLogBadgeEl.textContent = `✔ Logged to SQLite (#${data.detection_id})`;
            dbLogBadgeEl.className = "badge badge-success";
        } else if (data.is_low_confidence) {
            dbLogBadgeEl.textContent = "Low Confidence (Not Logged)";
            dbLogBadgeEl.className = "badge badge-warning";
        } else {
            dbLogBadgeEl.textContent = "Active (Duplicate Debounced)";
            dbLogBadgeEl.className = "badge badge-neutral";
        }

        // 6-Class Probability Bars
        if (data.class_probabilities) {
            Object.entries(data.class_probabilities).forEach(([cat, probVal]) => {
                const safeKey = cat.replace(/\s+/g, "-");
                const valEl = document.getElementById(`prob-val-${safeKey}`);
                const fillEl = document.getElementById(`prob-fill-${safeKey}`);
                if (valEl) valEl.textContent = `${Number(probVal).toFixed(1)}%`;
                if (fillEl) fillEl.style.width = `${Math.min(100, Math.max(0, probVal))}%`;
            });
        }
    }

    // ----------------------------------------------------------------------
    // 4. REFRESH DASHBOARD COUNTERS & RECENT TABLE
    // ----------------------------------------------------------------------

    async function refreshDashboardStats() {
        try {
            const [statsRes, detRes] = await Promise.all([
                fetch("/api/stats"),
                fetch("/api/detections?limit=10")
            ]);

            if (statsRes.ok) {
                const stats = await statsRes.json();
                const totalEl = document.getElementById("stat-total");
                const todayEl = document.getElementById("stat-today");
                if (totalEl) totalEl.textContent = stats.total_detections;
                if (todayEl) todayEl.textContent = stats.today_detections;

                if (stats.category_counts) {
                    Object.entries(stats.category_counts).forEach(([cat, count]) => {
                        const safeKey = cat.replace(/\s+/g, "-");
                        const countEl = document.getElementById(`stat-count-${safeKey}`);
                        if (countEl) countEl.textContent = count;
                    });
                }
            }

            if (detRes.ok) {
                const detData = await detRes.json();
                renderRecentDetectionsTable(detData.detections || []);
            }
        } catch (err) {
            console.error("Failed to refresh stats:", err);
        }
    }

    function renderRecentDetectionsTable(detections) {
        if (!recentTbodyEl) return;

        if (!detections.length) {
            recentTbodyEl.innerHTML = `
                <tr>
                    <td colspan="4" class="empty-table-msg">
                        No detections recorded yet. Start the camera or click a quick test sample above!
                    </td>
                </tr>
            `;
            return;
        }

        recentTbodyEl.innerHTML = detections
            .map((item) => {
                const timeOnly = item.timestamp.includes("T")
                    ? item.timestamp.split("T")[1]
                    : item.timestamp;
                return `
                    <tr>
                        <td class="td-time">${escapeHtml(timeOnly)}</td>
                        <td><span class="category-pill" data-category="${escapeHtml(item.waste_type)}">${escapeHtml(item.waste_type)}</span></td>
                        <td><strong>${Number(item.confidence).toFixed(1)}%</strong></td>
                        <td>${escapeHtml(item.recommended_bin)}</td>
                    </tr>
                `;
            })
            .join("");
    }

    // ----------------------------------------------------------------------
    // 5. SYNTHETIC SAMPLE CARD GENERATOR (FOR WORKSHOP QUICK TESTS)
    // ----------------------------------------------------------------------

    function renderPresetSampleCanvas(ctx, category) {
        const palettes = {
            "Plastic": { bg1: "#1e3a8a", bg2: "#3b82f6", label: "♻️ PLASTIC BOTTLE", sub: "PET-1 Recyclable Polymer" },
            "Paper": { bg1: "#f8fafc", bg2: "#e2e8f0", label: "📄 PAPER & CARDBOARD", sub: "Clean Cellulose Fiber", darkText: true },
            "Metal": { bg1: "#475569", bg2: "#94a3b8", label: "🥫 ALUMINUM CAN", sub: "Recyclable Metal Alloy" },
            "Glass": { bg1: "#083344", bg2: "#06b6d4", label: "🍾 GLASS BOTTLE", sub: "Clear Silicate Container" },
            "Organic": { bg1: "#14532d", bg2: "#22c55e", label: "🌱 ORGANIC WASTE", sub: "Biodegradable Fruit Peel" },
            "General Waste": { bg1: "#1e293b", bg2: "#475569", label: "🗑️ GENERAL WASTE", sub: "Non-Recyclable Wrapper" }
        };

        const info = palettes[category] || palettes["Plastic"];
        const grad = ctx.createLinearGradient(0, 0, 224, 224);
        grad.addColorStop(0, info.bg1);
        grad.addColorStop(1, info.bg2);
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 224, 224);

        // Add subtle geometric object silhouette in center
        ctx.fillStyle = info.darkText ? "rgba(15,23,42,0.08)" : "rgba(255,255,255,0.18)";
        ctx.beginPath();
        ctx.arc(112, 95, 48, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = info.darkText ? "#0f172a" : "#ffffff";
        ctx.font = "bold 15px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(info.label, 112, 165);

        ctx.font = "11px sans-serif";
        ctx.fillStyle = info.darkText ? "#334155" : "#e2e8f0";
        ctx.fillText(info.sub, 112, 185);
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    // ----------------------------------------------------------------------
    // 6. EVENT LISTENERS
    // ----------------------------------------------------------------------

    if (startCameraBtn) startCameraBtn.addEventListener("click", startCamera);
    if (stopCameraBtn) stopCameraBtn.addEventListener("click", stopCamera);

    if (placeholderStartBtn) {
        placeholderStartBtn.addEventListener("click", startCamera);
    }

    if (heroStartBtn) {
        heroStartBtn.addEventListener("click", () => {
            document.getElementById("scanner-section").scrollIntoView({ behavior: "smooth" });
            if (!mediaStream) startCamera();
        });
    }

    if (captureOnceBtn) {
        captureOnceBtn.addEventListener("click", () => {
            captureAndPredictFrame(true);
        });
    }

    if (intervalSelect) {
        intervalSelect.addEventListener("change", (e) => {
            captureIntervalMs = parseInt(e.target.value, 10) || 1200;
            if (mediaStream) {
                fpsBadgeEl.textContent = `Scanning (${(captureIntervalMs / 1000).toFixed(1)}s)`;
                startCaptureLoop();
            }
        });
    }

    // Image Upload Handler
    if (imageUploadInput) {
        imageUploadInput.addEventListener("change", async (e) => {
            const file = e.target.files && e.target.files[0];
            if (!file) return;

            if (mediaStream) {
                stopCamera();
            }

            const reader = new FileReader();
            reader.onload = (ev) => {
                uploadedPreviewEl.src = ev.target.result;
                uploadedPreviewEl.style.display = "block";
                videoEl.style.display = "none";
                placeholderEl.style.display = "none";
                reticleHintEl.textContent = `Uploaded: ${file.name}`;
            };
            reader.readAsDataURL(file);

            await sendImageBlobToApi(file, true, null);
            imageUploadInput.value = "";
        });
    }

    // Workshop Quick Test Preset Buttons
    document.querySelectorAll(".preset-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const category = btn.getAttribute("data-category");
            captureAndPredictFrame(true, category);
        });
    });

    // Refresh & Reset Database Buttons
    if (refreshStatsBtn) {
        refreshStatsBtn.addEventListener("click", refreshDashboardStats);
    }

    if (clearDbBtn) {
        clearDbBtn.addEventListener("click", async () => {
            if (!confirm("Reset all detection counters and clear the SQLite detection history?")) {
                return;
            }
            try {
                const res = await fetch("/api/detections", { method: "DELETE" });
                if (res.ok) {
                    await refreshDashboardStats();
                    dbLogBadgeEl.textContent = "History Reset";
                    dbLogBadgeEl.className = "badge badge-neutral";
                }
            } catch (err) {
                console.error("Failed to clear detections:", err);
            }
        });
    }
});
