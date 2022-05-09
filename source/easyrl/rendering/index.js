const express = require('express');
const app = express();
const http = require('http');
const server = http.createServer(app);
const { Server } = require("socket.io");
const io = new Server(server);

app.use(express.static(__dirname + "/public"));

app.get('/', (req, res) => {
    console.log("sending file");
    res.sendFile("/index.html");
});

io.on('connection', (socket) => {
    socket.on("sim", (sim) => {
        console.log(sim);
        socket.broadcast.emit("data", sim);

    });
});


server.listen(3000, "0.0.0.0", () => {
    console.log('listening on *:3000');
});
