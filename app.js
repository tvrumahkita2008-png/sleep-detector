const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctxOverlay = overlay.getContext('2d');
const processor = document.getElementById('processor');
const ctxProcessor = processor.getContext('2d', { willReadFrequently: true });
const statusPanel = document.getElementById('status-panel');
const logPanel = document.getElementById('log-panel');
const initBtn = document.getElementById('btn-init');

// --- AUDIO & NOTIFICATION STATE ---
const alarmSound = new Audio('https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg');
const successSound = new Audio('https://actions.google.com/sounds/v1/cartoon/wood_plank_flicks.ogg');
let currentState = "AWAITING"; 

let session;
const TARGET_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.4;

// Bind the boot sequence to the button
initBtn.addEventListener('click', async () => {
    initBtn.disabled = true;
    initBtn.innerText = "BOOTING...";
    
    // Request Web Notification Permissions
    if ("Notification" in window && Notification.permission !== "granted") {
        await Notification.requestPermission();
    }
    
    // Pre-load audio to bypass browser autoplay restrictions
    alarmSound.load();
    successSound.load();
    
    loadModel();
});

/**
 * Initialize the ONNX Runtime Session (High Performance)
 */
async function loadModel() {
    try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        
        // SPEED HACK 1: Unlock Multi-threading
        // This forces the browser to use multiple CPU cores instead of choking on just one.
        const numCores = navigator.hardwareConcurrency || 4;
        ort.env.wasm.numThreads = Math.min(4, numCores); 
        
        // SPEED HACK 2: Hardware Acceleration
        // We tell it to try 'webgl' (GPU) first. If the laptop GPU rejects it, it gracefully falls back to multi-threaded 'wasm'.
        session = await ort.InferenceSession.create('./best.onnx', { 
            executionProviders: ['webgl', 'wasm'],
            graphOptimizationLevel: 'all'
        });
        
        statusPanel.innerText = "STANDBY: AWAITING CAMERA INITIALIZATION";
        startCamera();
    } catch (e) {
        console.error("Model Mount Failure:", e);
        statusPanel.innerText = "SYSTEM FAILURE: UNABLE TO MOUNT MODEL";
        statusPanel.style.borderColor = "#ff0000";
    }
}

/**
 * Bind the hardware stream to the video element
 */
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        });
        
        video.srcObject = stream;
        video.onloadedmetadata = async () => {
            try {
                await video.play();
                statusPanel.innerText = "AWAITING SUBJECT...";
                initBtn.style.display = "none"; // Hide button once running
                requestAnimationFrame(processFrame);
            } catch (playError) {
                console.error("Playback failed:", playError);
            }
        };
    } catch (e) {
        console.error("Camera Access Failure:", e);
        statusPanel.innerText = "SYSTEM FAILURE: CAMERA ACCESS DENIED";
        statusPanel.style.borderColor = "#ff0000";
    }
}

/**
 * Extract frame, update UI log, and trigger Web Notification
 */
function logViolationAndNotify() {
    // 1. Capture the raw video frame
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const captureCtx = captureCanvas.getContext('2d');
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    // 2. Extract Data URL
    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.8);
    const timestamp = new Date().toLocaleTimeString();
    
    // 3. Inject into Sidebar Log
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <img src="${dataUrl}" alt="Violation Snapshot">
        <p>🚨 LOGGED: ${timestamp}</p>
    `;
    logPanel.insertBefore(entry, logPanel.children[1]);

    // 4. Trigger OS/Browser Notification
    if ("Notification" in window && Notification.permission === "granted") {
        new Notification("🚨 PROTOCOL VIOLATION", {
            body: `Unauthorized subject detected at ${timestamp}.`,
            icon: dataUrl, // Uses the captured face as the notification icon!
            vibrate: [200, 100, 200]
        });
    }
}

function calculateIoU(box1, box2) {
    const xA = Math.max(box1.x, box2.x);
    const yA = Math.max(box1.y, box2.y);
    const xB = Math.min(box1.x + box1.w, box2.x + box2.w);
    const yB = Math.min(box1.y + box1.h, box2.y + box2.h);
    const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    return intersectionArea / ((box1.w * box1.h) + (box2.w * box2.h) - intersectionArea);
}

function nonMaxSuppression(boxes, iouThreshold) {
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    while (boxes.length > 0) {
        const current = boxes.shift();
        result.push(current);
        boxes = boxes.filter(box => calculateIoU(current, box) < iouThreshold);
    }
    return result;
}

/**
 * Main Inference Loop
 */
async function processFrame() {
    if (!session) return;

    ctxProcessor.drawImage(video, 0, 0, TARGET_SIZE, TARGET_SIZE);
    const imageData = ctxProcessor.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE).data;

    const float32Data = new Float32Array(3 * TARGET_SIZE * TARGET_SIZE);
    for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {
        float32Data[i]                                   = imageData[i * 4] / 255.0;
        float32Data[i + TARGET_SIZE * TARGET_SIZE]       = imageData[i * 4 + 1] / 255.0;
        float32Data[i + 2 * TARGET_SIZE * TARGET_SIZE]   = imageData[i * 4 + 2] / 255.0;
    }

    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, TARGET_SIZE, TARGET_SIZE]);
    const results = await session.run({ [session.inputNames[0]]: inputTensor });
    const output = results[session.outputNames[0]].data; 
    
    let rawBoxes = [];
    const elements = 8400;

    for (let i = 0; i < elements; i++) {
        let x = output[i];
        let y = output[i + elements];
        let w = output[i + 2 * elements];
        let h = output[i + 3 * elements];
        const scoreMask = output[i + 4 * elements]; 
        const scoreNoMask = output[i + 5 * elements];
        const maxScore = Math.max(scoreMask, scoreNoMask);

        if (maxScore > CONFIDENCE_THRESHOLD) {
            if (w <= 1.5 && h <= 1.5) {
                x *= TARGET_SIZE; y *= TARGET_SIZE; w *= TARGET_SIZE; h *= TARGET_SIZE;
            }
            rawBoxes.push({
                x: x - w / 2, y: y - h / 2, w: w, h: h,
                score: maxScore,
                classId: scoreNoMask > scoreMask ? 1 : 0
            });
        }
    }

    const finalBoxes = nonMaxSuppression(rawBoxes, IOU_THRESHOLD);
    ctxOverlay.clearRect(0, 0, overlay.width, overlay.height);
    let isViolating = false;

    if (finalBoxes.length > 0) {
        finalBoxes.forEach(box => {
            const scaleX = overlay.width / TARGET_SIZE;
            const scaleY = overlay.height / TARGET_SIZE;
            const scaledX = box.x * scaleX;
            const scaledY = box.y * scaleY;
            const scaledW = box.w * scaleX;
            const scaledH = box.h * scaleY;

            if (box.classId === 1) isViolating = true;
            
            const color = box.classId === 1 ? '#FF3B30' : '#34C759'; 
            const labelText = box.classId === 1 ? `NO MASK ${(box.score * 100).toFixed(1)}%` : `MASK ${(box.score * 100).toFixed(1)}%`;

            ctxOverlay.strokeStyle = color;
            ctxOverlay.lineWidth = 4;
            ctxOverlay.strokeRect(scaledX, scaledY, scaledW, scaledH);
            
            ctxOverlay.font = 'bold 18px monospace';
            const textWidth = ctxOverlay.measureText(labelText).width;
            ctxOverlay.fillStyle = color;
            ctxOverlay.fillRect(scaledX - 2, scaledY - 28, textWidth + 12, 28);
            
            ctxOverlay.fillStyle = '#FFFFFF';
            ctxOverlay.fillText(labelText, scaledX + 4, scaledY - 8);
        });

        // --- GLOBAL STATE MACHINE ---
        if (isViolating) {
            statusPanel.innerText = "🚨 ACCESS DENIED: PROTOCOL VIOLATED";
            statusPanel.style.backgroundColor = "#4a0000";
            statusPanel.style.borderColor = "#FF3B30";
            statusPanel.style.color = "#ffcccc";
            
            if (currentState !== "DENIED") {
                alarmSound.currentTime = 0; 
                alarmSound.play();
                logViolationAndNotify(); // Snapshot & Web Notification!
                currentState = "DENIED";
            }
        } else {
            statusPanel.innerText = "✅ ACCESS GRANTED: PROCEED TO AIRLOCK";
            statusPanel.style.backgroundColor = "#003300";
            statusPanel.style.borderColor = "#34C759";
            statusPanel.style.color = "#ccffcc";
            
            if (currentState !== "GRANTED") {
                successSound.currentTime = 0; 
                successSound.play();
                currentState = "GRANTED";
            }
        }
    } else {
        statusPanel.innerText = "AWAITING SUBJECT...";
        statusPanel.style.backgroundColor = "transparent";
        statusPanel.style.borderColor = "#555";
        statusPanel.style.color = "#ffffff";
        currentState = "AWAITING"; 
    }

    requestAnimationFrame(processFrame);
}
