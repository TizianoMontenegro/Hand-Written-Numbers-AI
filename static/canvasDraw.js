const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let drawing = false;

// Initialize canvas with white background
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "black";
ctx.lineWidth = 20;
ctx.lineCap = "round";

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseout", stopDrawing);

function startDrawing(e) {
    drawing = true;
    draw(e);  // Start drawing immediately
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath();  // Start new path for next stroke
}

function draw(e) {
    if (!drawing) return;
    
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

const clearCanvasButton = document.getElementById("clearCanvasButton")
const predictButton = document.getElementById("predictButton")

function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    document.getElementById("result").textContent = "None";
}

async function predict() {
    const dataURL = canvas.toDataURL("image/png");  // Convert canvas to base64
    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL })
    });
    const result = await response.json();
    document.getElementById("result").textContent = result.digit;
}

clearCanvasButton.addEventListener("click", clearCanvas)
predictButton.addEventListener("click", predict)