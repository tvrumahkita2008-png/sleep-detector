const btnInit = document.getElementById("btn-init");
const statusPanel = document.getElementById("status-panel");
const video = document.getElementById("webcam");

btnInit.addEventListener("click", async () => {
    try {
        statusPanel.innerText = "BOOTING...";
        btnInit.disabled = true;

        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        video.srcObject = stream;

        statusPanel.innerText = "SYSTEM ONLINE";
        statusPanel.style.borderColor = "lime";
        statusPanel.style.color = "lime";

    } catch (err) {
        console.log(err);

        statusPanel.innerText = "SYSTEM FAILURE: CAMERA ACCESS DENIED";
        statusPanel.style.borderColor = "red";
        statusPanel.style.color = "red";
    }
});
