// frontend/script.js

let scene, camera, renderer;
let balls = [];
let trajectory = [];
let frame = 0;
const numBalls = 5;
const radius = 0.03;
const totalFrames = 500;
const animationSpeed = 1;

init();
loadTrajectory();
animate();

function init() {
  scene = new THREE.Scene();
  camera = new THREE.OrthographicCamera(0, 1, 1, 0, 0.1, 10);
  camera.position.z = 1;

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0xffffff);  // 흰색 배경
  document.body.appendChild(renderer.domElement);

  for (let i = 0; i < numBalls; i++) {
    const geometry = new THREE.CircleGeometry(radius, 32);
    const material = new THREE.MeshBasicMaterial({ color: 0x00ccff });
    const circle = new THREE.Mesh(geometry, material);
    scene.add(circle);
    balls.push(circle);
  }

  document.getElementById("refreshBtn").addEventListener("click", refreshTrajectory);
}

function loadTrajectory() {
  fetch("http://localhost:8000/trajectory", { cache: "no-store" })
    .then(response => response.json())
    .then(data => {
      trajectory = data;
      frame = 0; // 리셋
    });
}

function refreshTrajectory() {
  fetch("http://localhost:8000/trajectory/refresh", { method: "POST" })
    .then(() => loadTrajectory());
}

function animate() {
  requestAnimationFrame(animate);

  if (trajectory.length > 0) {
    const currentFrame = trajectory[frame % trajectory.length];

    for (let i = 0; i < numBalls; i++) {
      const [x, y] = currentFrame[i];
      balls[i].position.set(x, y, 0);
    }

    frame += animationSpeed;
  }

  renderer.render(scene, camera);
}
