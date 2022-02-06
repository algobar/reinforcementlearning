
const socket = io("http://172.31.182.228:3000");

socket.on("data", (data) => {
    console.log("got data");
    canvas = document.getElementById("visualizer");
    ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, 300, 300); // clear canvas
    for (var key in data) {
        ctx.beginPath();
        ctx.fillRect(data[key].x * 15, data[key].y * 15, 10, 10);
        ctx.stroke();

    }
});

