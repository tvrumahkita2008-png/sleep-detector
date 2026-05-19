// script.js

const video = document.getElementById("video");
const statusText = document.getElementById("status");

// Menghubungkan kamera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });

    video.srcObject = stream;

    statusText.innerHTML = "Kamera aktif ✅";

  } catch (error) {
    statusText.innerHTML = "Kamera gagal diakses ❌";
    console.error(error);
  }
}

startCamera();

// Simulasi sleep detection
setInterval(() => {

  // angka random simulasi mata tertutup
  let sleepy = Math.random();

  if (sleepy > 0.7) {
    statusText.innerHTML = "MENGANTUK 😴";
    statusText.style.color = "red";
  } else {
    statusText.innerHTML = "NORMAL 😀";
    statusText.style.color = "lime";
  }

}, 3000);
