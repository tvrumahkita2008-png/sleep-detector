const CONFIG = {
    modelPath: "./best.onnx",
    labels: ["awake", "sleeping"],
    threshold: 0.45,
    iouThreshold: 0.4
};

const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctxOverlay = overlay.getContext("2d");

const processor = document.getElementById("processor");
const ctxProcessor = processor.getContext("2d");

const status = document.getElementById("status");
const button = document.getElementById("btn-init");

const TARGET_SIZE = 640;

let session;
let isRunning = false;

// ===================== LOAD MODEL =====================
button.addEventListener("click", async () => {
    try {
        button.innerText = "MEMUAT AI...";
        button.disabled = true;

        ort.env.wasm.wasmPaths =
            "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        session = await ort.InferenceSession.create(CONFIG.modelPath, {
            executionProviders: ["webgl", "wasm"]
        });

        status.innerText = "MODEL LOADED";
        startCamera();

    } catch (err) {
        console.error(err);
        status.innerText = "GAGAL LOAD MODEL";
        button.disabled = false;
    }
});

// ===================== CAMERA =====================
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false
        });

        video.srcObject = stream;

        video.onloadedmetadata = () => {
            video.play();
            button.style.display = "none";
            status.innerText = "AI AKTIF";

            isRunning = true;
            processFrame();
        };

    } catch (err) {
        console.error(err);
        status.innerText = "KAMERA ERROR / IZIN DITOLAK";
    }
}

// ===================== LOOP =====================
async function processFrame() {
    if (!session || !isRunning) return;

    ctxProcessor.drawImage(video, 0, 0, TARGET_SIZE, TARGET_SIZE);

    const imageData = ctxProcessor.getImageData(
        0, 0, TARGET_SIZE, TARGET_SIZE
    ).data;

    const input = new Float32Array(3 * TARGET_SIZE * TARGET_SIZE);

    for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {
        input[i] =
            imageData[i * 4] / 255;

        input[i + TARGET_SIZE * TARGET_SIZE] =
            imageData[i * 4 + 1] / 255;

        input[i + 2 * TARGET_SIZE * TARGET_SIZE] =
            imageData[i * 4 + 2] / 255;
    }

    const tensor = new ort.Tensor(
        "float32",
        input,
        [1, 3, TARGET_SIZE, TARGET_SIZE]
    );

    const results = await session.run({
        [session.inputNames[0]]: tensor
    });

    const output = results[session.outputNames[0]].data;

    const elements = 8400;
    const numClasses = CONFIG.labels.length;

    let boxes = [];

    for (let i = 0; i < elements; i++) {

        let maxScore = 0;
        let classId = -1;

        for (let c = 0; c < numClasses; c++) {
            const score = output[i + (4 + c) * elements];

            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        if (maxScore > CONFIG.threshold) {

            let x = output[i];
            let y = output[i + elements];
            let w = output[i + 2 * elements];
            let h = output[i + 3 * elements];

            if (w <= 1.5) {
                x *= TARGET_SIZE;
                y *= TARGET_SIZE;
                w *= TARGET_SIZE;
                h *= TARGET_SIZE;
            }

            boxes.push({
                x: x - w / 2,
                y: y - h / 2,
                w,
                h,
                score: maxScore,
                classId
            });
        }
    }

    drawBoxes(boxes);

    requestAnimationFrame(processFrame);
}

// ===================== DRAW =====================
function drawBoxes(boxes) {
    ctxOverlay.clearRect(0, 0, overlay.width, overlay.height);

    const scaleX = overlay.width / TARGET_SIZE;
    const scaleY = overlay.height / TARGET_SIZE;

    let isSleeping = false;

    boxes.forEach(box => {

        const label = CONFIG.labels[box.classId];

        let color = "#34C759";

        if (label === "sleeping") {
            color = "#FF3B30";
            isSleeping = true;
        }

        ctxOverlay.strokeStyle = color;
        ctxOverlay.lineWidth = 3;

        ctxOverlay.strokeRect(
            box.x * scaleX,
            box.y * scaleY,
            box.w * scaleX,
            box.h * scaleY
        );

        ctxOverlay.fillStyle = color;
        ctxOverlay.font = "bold 18px Arial";

        ctxOverlay.fillText(
            `${label} ${(box.score * 100).toFixed(0)}%`,
            box.x * scaleX,
            box.y * scaleY - 5
        );
    });

    // STATUS GLOBAL (lebih stabil)
    if (isSleeping) {
        status.innerText = "😴 TERDETEKSI TIDUR";
    } else {
        status.innerText = "😃 TERJAGA";
    }
}
