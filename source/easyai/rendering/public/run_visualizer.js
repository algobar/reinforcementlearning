
//import Two from 'https://cdn.skypack.dev/two.js@latest';

const socket = io("http://localhost:3000");

socket.on("data", (data) => {
    console.log("got data");
    canvas = document.getElementById("visualizer");
    ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, 300, 300); // clear canvas
    for (var key in data) {
        ctx.beginPath();
        ctx.fillRect(data[key].x * 10 - 5, data[key].y * 10 - 5, 10, 10);
        ctx.stroke();

    }
});

