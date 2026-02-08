let theShader;
let startTime;

// Variables pour contrôler la caméra
let rotationX = 0;
let rotationY = 0;
let distance = 3.0;
let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;

function preload() {
  theShader = loadShader(
    '/assets/js/pass.vert',
    '/assets/js/mandelbub_3d.frag'
  );
}

function setup() {
  createCanvas(windowWidth, windowHeight, WEBGL);
  noStroke();
  startTime = millis();
}

function draw() {
  shader(theShader);
  theShader.setUniform('uTime', (millis() - startTime) / 1000.0);
  theShader.setUniform('uResolution', [width, height]);
  theShader.setUniform('uPixelDensity', pixelDensity());
  
  // Envoyer les deux angles de rotation au shader
  theShader.setUniform('uRotAngleY', rotationY);
  theShader.setUniform('uRotAngleX', rotationX);
  theShader.setUniform('uDistance', distance);

  rect(-width / 2, -height / 2, width, height);
}

function mousePressed() {
  isDragging = true;
  lastMouseX = mouseX;
  lastMouseY = mouseY;
}

function mouseReleased() {
  isDragging = false;
}

function mouseDragged() {
  if (isDragging) {
    // Calculer le déplacement de la souris
    let deltaX = mouseX - lastMouseX;
    let deltaY = mouseY - lastMouseY;
    
    // Mettre à jour les rotations
    rotationY += deltaX * 0.01; // Rotation horizontale (autour de Y)
    rotationX += deltaY * 0.01; // Rotation verticale (autour de X)
    
    // Limiter la rotation verticale pour éviter le retournement
    rotationX = constrain(rotationX, -Math.PI / 2, Math.PI / 2);
    
    // Sauvegarder la position actuelle
    lastMouseX = mouseX;
    lastMouseY = mouseY;
  }
}

function mouseWheel(event) {
  // Zoomer avec la molette de la souris
  distance += event.delta * 0.001;
  
  // Limiter le zoom
  distance = constrain(distance, 1.5, 10.0);
  
  // Empêcher le défilement de la page
  return false;
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
