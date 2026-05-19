const btnInit = document.getElementById("btn-init");
const statusPanel = document.getElementById("status-panel");
const video = document.getElementById("webcam");

const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const logPanel = document.getElementById("log-panel");

let alarmPlayed = false;

// suara alarm
const alarm = new Audio("alarm.mp3");

btnInit.addEventListener("click", async () => {
    try {

        statusPanel.innerText = "BOOTING...";
        btnInit.disabled = true;

        // akses kamera
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        video.srcObject = stream;

        statusPanel.innerText = "SYSTEM ONLINE";
        statusPanel.style.borderColor = "lime";
        statusPanel.style.color = "lime";

        detectSleep();

    } catch (err) {

        console.log(err);

        statusPanel.innerText = "SYSTEM FAILURE: CAMERA ACCESS DENIED";
        statusPanel.style.borderColor = "red";
        statusPanel.style.color = "red";
    }
});

// simulasi sleep detection
function detectSleep() {

    setInterval(() => {

        // simulasi random tidur
        const sleepDetected = Math.random() < 0.3;

        ctx.clearRect(0, 0, overlay.width, overlay.height);

        if (sleepDetected) {

            // kotak merah
            ctx.strokeStyle = "red";
            ctx.lineWidth = 4;
            ctx.strokeRect(180, 80, 280, 300);

            // tulisan
            ctx.fillStyle = "red";
            ctx.font = "30px monospace";
            ctx.fillText("DROWSY!", 240, 60);

            // status
            statusPanel.innerText = "WARNING: DRIVER DROWSINESS DETECTED";
            statusPanel.style.borderColor = "red";
            statusPanel.style.color = "red";

            // bunyi alarm
            if (!alarmPlayed) {
                alarm.play();
                alarmPlayed = true;
            }

            // simpan log
            saveLog();

        } else {

            statusPanel.innerText = "SYSTEM ONLINE";
            statusPanel.style.borderColor = "lime";
            statusPanel.style.color = "lime";

            alarm.pause();
            alarm.currentTime = 0;

            alarmPlayed = false;
        }

    }, 3000);
}

// log screenshot
function saveLog() {

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const c = canvas.getContext("2d");

    c.drawImage(video, 0, 0);

    const imgData = canvas.toDataURL("image/png");

    const log = document.createElement("div");
    log.className = "log-entry";

    log.innerHTML = `
        <img src="${imgData}">
        <p>Drowsiness Detected</p>
    `;

    logPanel.prepend(log);
}
