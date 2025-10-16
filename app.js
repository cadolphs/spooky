// DOM elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const status = document.getElementById('status');
const filterBtns = document.querySelectorAll('.filter-btn');

// State
let stream = null;
let animationId = null;
let currentFilter = 'none';
let startTime = 0; // For time-based effects

// Face detection state
let faceDetector = null;
let currentFaceLandmarks = null;
let previousFaceLandmarks = null; // For smoothing
let frameCount = 0; // For throttling face detection
let eyePositionHistory = []; // Trail history for swirl effect
const MAX_EYE_HISTORY = 12; // Number of trail positions to keep

// WebGL state
let gl = null;
let edgeDetectionProgram = null;
let jitterProgram = null;
let blurProgram = null;
let compositeProgram = null;
let videoTexture = null;
let edgeTexture = null;
let jitteredEdgeTexture = null;
let blurredEdgeTexture = null;
let framebuffer = null;
let framebuffer2 = null;
let framebuffer3 = null;
let positionBuffer = null;
let texCoordBuffer = null;

// ============================================================================
// WEBGL INFRASTRUCTURE
// ============================================================================

/**
 * Initialize WebGL context and shaders
 */
function initWebGL() {
    // Get WebGL context from canvas
    gl = canvas.getContext('webgl', {
        premultipliedAlpha: false,
        preserveDrawingBuffer: true
    });

    if (!gl) {
        console.error('WebGL not supported');
        return false;
    }

    console.log('✓ WebGL context created');

    // Vertex shader - simple full-screen quad
    const vertexShaderSource = `
        attribute vec2 a_position;
        attribute vec2 a_texCoord;
        varying vec2 v_texCoord;

        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
            v_texCoord = a_texCoord;
        }
    `;

    // Fragment shader for edge detection (Laplace kernel)
    const edgeDetectionFragmentSource = `
        precision mediump float;
        uniform sampler2D u_image;
        uniform vec2 u_textureSize;
        varying vec2 v_texCoord;

        void main() {
            vec2 onePixel = 1.0 / u_textureSize;

            // Laplace kernel (3x3)
            vec4 sum = vec4(0.0);
            sum += texture2D(u_image, v_texCoord + vec2(-onePixel.x, -onePixel.y)) * 1.0;
            sum += texture2D(u_image, v_texCoord + vec2(0.0, -onePixel.y)) * 1.0;
            sum += texture2D(u_image, v_texCoord + vec2(onePixel.x, -onePixel.y)) * 1.0;
            sum += texture2D(u_image, v_texCoord + vec2(-onePixel.x, 0.0)) * 1.0;
            sum += texture2D(u_image, v_texCoord) * -8.0;
            sum += texture2D(u_image, v_texCoord + vec2(onePixel.x, 0.0)) * 1.0;
            sum += texture2D(u_image, v_texCoord + vec2(-onePixel.x, onePixel.y)) * 1.0;
            sum += texture2D(u_image, v_texCoord + vec2(0.0, onePixel.y)) * 1.0;
            sum += texture2D(u_image, v_texCoord + vec2(onePixel.x, onePixel.y)) * 1.0;

            gl_FragColor = vec4(abs(sum.rgb), 1.0);
        }
    `;

    // Fragment shader for jitter (like addJitter in Haunted filter)
    const jitterFragmentSource = `
        precision mediump float;
        uniform sampler2D u_image;
        uniform vec2 u_textureSize;
        uniform float u_time;
        varying vec2 v_texCoord;

        // Noise function
        float noise(vec2 co) {
            return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
        }

        void main() {
            vec2 onePixel = 1.0 / u_textureSize;
            vec2 pixelCoord = v_texCoord * u_textureSize;

            // Random offset based on pixel position and time (intensity = 3 pixels like Haunted)
            float offsetX = (noise(pixelCoord * 0.1 + u_time * 10.0) - 0.5) * 2.0 * 3.0;
            float offsetY = (noise(pixelCoord * 0.1 + u_time * 10.0 + 100.0) - 0.5) * 2.0 * 3.0;

            vec2 sourceCoord = (pixelCoord + vec2(offsetX, offsetY)) / u_textureSize;

            // Clamp to texture bounds
            sourceCoord = clamp(sourceCoord, vec2(0.0), vec2(1.0));

            gl_FragColor = texture2D(u_image, sourceCoord);
        }
    `;

    // Fragment shader for Gaussian blur (thicker edges - 20 pixel radius)
    const blurFragmentSource = `
        precision mediump float;
        uniform sampler2D u_image;
        uniform vec2 u_textureSize;
        uniform vec2 u_direction;
        varying vec2 v_texCoord;

        void main() {
            vec2 onePixel = 1.0 / u_textureSize;

            // 9-tap Gaussian blur kernel with 20 pixel radius for thicker edges
            // Weights: [0.05, 0.09, 0.12, 0.15, 0.16, 0.15, 0.12, 0.09, 0.05]
            vec4 sum = vec4(0.0);
            sum += texture2D(u_image, v_texCoord + u_direction * onePixel * -10.0) * 0.05;
            sum += texture2D(u_image, v_texCoord + u_direction * onePixel * -7.5) * 0.09;
            sum += texture2D(u_image, v_texCoord + u_direction * onePixel * -5.0) * 0.12;
            sum += texture2D(u_image, v_texCoord + u_direction * onePixel * -2.5) * 0.15;
            sum += texture2D(u_image, v_texCoord) * 0.16;
            sum += texture2D(u_image, v_texCoord + u_direction * onePixel * 2.5) * 0.15;
            sum += texture2D(u_image, v_texCoord + u_direction * onePixel * 5.0) * 0.12;
            sum += texture2D(u_image, v_texCoord + u_direction * onePixel * 7.5) * 0.09;
            sum += texture2D(u_image, v_texCoord + u_direction * onePixel * 10.0) * 0.05;

            gl_FragColor = sum;
        }
    `;

    // Fragment shader for final composite (grayscale + invert + morphing colored edge glow)
    const compositeFragmentSource = `
        precision mediump float;
        uniform sampler2D u_image;
        uniform sampler2D u_edges;
        uniform float u_time;
        uniform vec2 u_textureSize;
        varying vec2 v_texCoord;

        // Noise function for jitter
        float noise(vec2 co) {
            return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
        }

        void main() {
            vec4 color = texture2D(u_image, v_texCoord);

            // Apply jitter when sampling edges (AFTER blur, so blur doesn't smooth it out)
            vec2 pixelCoord = v_texCoord * u_textureSize;
            vec2 onePixel = 1.0 / u_textureSize;
            float offsetX = (noise(pixelCoord * 0.1 + u_time * 10.0) - 0.5) * 2.0 * 3.0;
            float offsetY = (noise(pixelCoord * 0.1 + u_time * 10.0 + 100.0) - 0.5) * 2.0 * 3.0;
            vec2 jitteredCoord = v_texCoord + vec2(offsetX, offsetY) * onePixel;

            vec4 edge = texture2D(u_edges, jitteredCoord);

            // Convert to grayscale
            float gray = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;

            // Invert for ghostly effect
            float inverted = 1.0 - gray;
            vec3 result = vec3(inverted);

            // Time-based color drift (multiple sine waves for organic morphing) - MUCH MORE COLORFUL
            float redDrift = sin(u_time * 0.3) * 0.5;
            float greenDrift = sin(u_time * 0.23 + 1.5) * 0.6;
            float blueDrift = sin(u_time * 0.17 + 3.0) * 0.7;

            // Add spatial variation for more interesting colors
            float spatialVar = sin(v_texCoord.x * 3.14159 * 2.0 + u_time * 0.5) *
                              cos(v_texCoord.y * 3.14159 * 2.0 + u_time * 0.3);

            // Add glowing blurred edge overlay with morphing colors
            float edgeIntensity = max(edge.r, max(edge.g, edge.b));
            float threshold = 5.0 / 255.0; // Lower threshold for thicker edges

            if (edgeIntensity > threshold) {
                float glowStrength = min(edgeIntensity * 2.0, 1.0); // Amplify even more

                // Morphing rainbow edge glow
                result.r = min(1.0, result.r + glowStrength * (0.3 + redDrift + spatialVar * 0.3));
                result.g = min(1.0, result.g + glowStrength * (0.5 + greenDrift + spatialVar * 0.4));
                result.b = min(1.0, result.b + glowStrength * (0.8 + blueDrift + spatialVar * 0.5));
            } else {
                // Subtle drift on base image too
                result.r = clamp(result.r + redDrift * 0.1, 0.0, 1.0);
                result.g = clamp(result.g + greenDrift * 0.1, 0.0, 1.0);
                result.b = clamp(result.b + blueDrift * 0.1, 0.0, 1.0);
            }

            gl_FragColor = vec4(result, 1.0);
        }
    `;

    // Compile shaders and create programs
    edgeDetectionProgram = createProgram(gl, vertexShaderSource, edgeDetectionFragmentSource);
    jitterProgram = createProgram(gl, vertexShaderSource, jitterFragmentSource);
    blurProgram = createProgram(gl, vertexShaderSource, blurFragmentSource);
    compositeProgram = createProgram(gl, vertexShaderSource, compositeFragmentSource);

    if (!edgeDetectionProgram || !jitterProgram || !blurProgram || !compositeProgram) {
        console.error('Failed to create shader programs');
        return false;
    }

    // Create buffers for full-screen quad
    positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1,
         1, -1,
        -1,  1,
        -1,  1,
         1, -1,
         1,  1
    ]), gl.STATIC_DRAW);

    texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        0, 1,
        1, 1,
        0, 0,
        0, 0,
        1, 1,
        1, 0
    ]), gl.STATIC_DRAW);

    // Create textures
    videoTexture = gl.createTexture();
    edgeTexture = gl.createTexture();
    jitteredEdgeTexture = gl.createTexture();
    blurredEdgeTexture = gl.createTexture();

    // Set up edge texture
    gl.bindTexture(gl.TEXTURE_2D, edgeTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Set up jittered edge texture
    gl.bindTexture(gl.TEXTURE_2D, jitteredEdgeTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Set up blurred edge texture
    gl.bindTexture(gl.TEXTURE_2D, blurredEdgeTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Create framebuffers for multi-pass rendering
    framebuffer = gl.createFramebuffer();
    framebuffer2 = gl.createFramebuffer();
    framebuffer3 = gl.createFramebuffer();

    console.log('✓ WebGL shaders and buffers initialized');
    return true;
}

/**
 * Helper: Compile shader
 */
function compileShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }

    return shader;
}

/**
 * Helper: Create shader program
 */
function createProgram(gl, vertexSource, fragmentSource) {
    const vertexShader = compileShader(gl, vertexSource, gl.VERTEX_SHADER);
    const fragmentShader = compileShader(gl, fragmentSource, gl.FRAGMENT_SHADER);

    if (!vertexShader || !fragmentShader) {
        return null;
    }

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program linking error:', gl.getProgramInfoLog(program));
        gl.deleteProgram(program);
        return null;
    }

    return program;
}

/**
 * Render Ghost Head filter using WebGL (GPU-accelerated)
 */
function renderGhostHeadWebGL() {
    if (!gl || !edgeDetectionProgram || !jitterProgram || !blurProgram || !compositeProgram) {
        console.error('WebGL not initialized');
        return;
    }

    const width = canvas.width;
    const height = canvas.height;

    // Update video texture with current canvas content
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);

    // Set up texture sizes
    gl.bindTexture(gl.TEXTURE_2D, edgeTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    gl.bindTexture(gl.TEXTURE_2D, jitteredEdgeTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    gl.bindTexture(gl.TEXTURE_2D, blurredEdgeTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    const elapsed = (performance.now() - startTime) / 1000.0;

    // PASS 1: Edge detection
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, edgeTexture, 0);
    gl.viewport(0, 0, width, height);

    gl.useProgram(edgeDetectionProgram);

    const edgePosLoc = gl.getAttribLocation(edgeDetectionProgram, 'a_position');
    const edgeTexLoc = gl.getAttribLocation(edgeDetectionProgram, 'a_texCoord');

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(edgePosLoc);
    gl.vertexAttribPointer(edgePosLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.enableVertexAttribArray(edgeTexLoc);
    gl.vertexAttribPointer(edgeTexLoc, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.uniform1i(gl.getUniformLocation(edgeDetectionProgram, 'u_image'), 0);
    gl.uniform2f(gl.getUniformLocation(edgeDetectionProgram, 'u_textureSize'), width, height);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // PASS 2: Jitter the edge pixels (distort the edges themselves)
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer2);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, jitteredEdgeTexture, 0);
    gl.viewport(0, 0, width, height);

    gl.useProgram(jitterProgram);

    const jitterPosLoc = gl.getAttribLocation(jitterProgram, 'a_position');
    const jitterTexLoc = gl.getAttribLocation(jitterProgram, 'a_texCoord');

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(jitterPosLoc);
    gl.vertexAttribPointer(jitterPosLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.enableVertexAttribArray(jitterTexLoc);
    gl.vertexAttribPointer(jitterTexLoc, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, edgeTexture);
    gl.uniform1i(gl.getUniformLocation(jitterProgram, 'u_image'), 0);
    gl.uniform2f(gl.getUniformLocation(jitterProgram, 'u_textureSize'), width, height);
    gl.uniform1f(gl.getUniformLocation(jitterProgram, 'u_time'), elapsed);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // PASS 3: Horizontal blur (on jittered edges)
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer3);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, blurredEdgeTexture, 0);
    gl.viewport(0, 0, width, height);

    gl.useProgram(blurProgram);

    const blurPosLoc1 = gl.getAttribLocation(blurProgram, 'a_position');
    const blurTexLoc1 = gl.getAttribLocation(blurProgram, 'a_texCoord');

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(blurPosLoc1);
    gl.vertexAttribPointer(blurPosLoc1, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.enableVertexAttribArray(blurTexLoc1);
    gl.vertexAttribPointer(blurTexLoc1, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, jitteredEdgeTexture);
    gl.uniform1i(gl.getUniformLocation(blurProgram, 'u_image'), 0);
    gl.uniform2f(gl.getUniformLocation(blurProgram, 'u_textureSize'), width, height);
    gl.uniform2f(gl.getUniformLocation(blurProgram, 'u_direction'), 1.0, 0.0); // Horizontal

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // PASS 4: Vertical blur
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, edgeTexture, 0);
    gl.viewport(0, 0, width, height);

    gl.useProgram(blurProgram);

    const blurPosLoc2 = gl.getAttribLocation(blurProgram, 'a_position');
    const blurTexLoc2 = gl.getAttribLocation(blurProgram, 'a_texCoord');

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(blurPosLoc2);
    gl.vertexAttribPointer(blurPosLoc2, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.enableVertexAttribArray(blurTexLoc2);
    gl.vertexAttribPointer(blurTexLoc2, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, blurredEdgeTexture);
    gl.uniform1i(gl.getUniformLocation(blurProgram, 'u_image'), 0);
    gl.uniform2f(gl.getUniformLocation(blurProgram, 'u_textureSize'), width, height);
    gl.uniform2f(gl.getUniformLocation(blurProgram, 'u_direction'), 0.0, 1.0); // Vertical

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // PASS 5: Composite (grayscale + invert + jittered blurred edge glow with morphing colors)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);

    gl.useProgram(compositeProgram);

    // Set up attributes
    const compPosLoc = gl.getAttribLocation(compositeProgram, 'a_position');
    const compTexLoc = gl.getAttribLocation(compositeProgram, 'a_texCoord');

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.enableVertexAttribArray(compPosLoc);
    gl.vertexAttribPointer(compPosLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.enableVertexAttribArray(compTexLoc);
    gl.vertexAttribPointer(compTexLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.uniform1i(gl.getUniformLocation(compositeProgram, 'u_image'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, edgeTexture); // Now contains blurred edges
    gl.uniform1i(gl.getUniformLocation(compositeProgram, 'u_edges'), 1);

    // Pass time for color morphing and texture size for jitter
    gl.uniform1f(gl.getUniformLocation(compositeProgram, 'u_time'), elapsed);
    gl.uniform2f(gl.getUniformLocation(compositeProgram, 'u_textureSize'), width, height);

    // Draw
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Now read back to 2D context for eye trail rendering
    const imageData = ctx.getImageData(0, 0, width, height);

    // Draw rainbow eye trails (same as CPU version)
    if (eyePositionHistory.length > 0) {
        const data = imageData.data;
        const currentTime = performance.now();

        eyePositionHistory.forEach((historyItem, index) => {
            const ageRatio = index / eyePositionHistory.length;
            const opacity = ageRatio;
            const radius = 25 + (10 * ageRatio);
            const spiralOffset = (1 - ageRatio) * 10;
            const spiralAngle = ageRatio * Math.PI * 2;

            [historyItem.left, historyItem.right].forEach(eye => {
                const eyeX = eye.x + Math.cos(spiralAngle) * spiralOffset;
                const eyeY = eye.y + Math.sin(spiralAngle) * spiralOffset;

                for (let dy = -radius; dy <= radius; dy++) {
                    for (let dx = -radius; dx <= radius; dx++) {
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist <= radius) {
                            const px = Math.floor(eyeX + dx);
                            const py = Math.floor(eyeY + dy);
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                const idx = (py * width + px) * 4;
                                const glowStrength = Math.pow(1 - (dist / radius), 2) * opacity;

                                const currentR = data[idx];
                                const currentG = data[idx + 1];
                                const currentB = data[idx + 2];

                                // Rainbow color calculation
                                const hue = ageRatio;
                                let r, g, b;

                                if (hue < 0.17) {
                                    const t = hue / 0.17;
                                    r = 255 * (1 - t);
                                    g = 0;
                                    b = 255;
                                } else if (hue < 0.34) {
                                    const t = (hue - 0.17) / 0.17;
                                    r = 0;
                                    g = 255 * t;
                                    b = 255;
                                } else if (hue < 0.5) {
                                    const t = (hue - 0.34) / 0.16;
                                    r = 0;
                                    g = 255;
                                    b = 255 * (1 - t);
                                } else if (hue < 0.67) {
                                    const t = (hue - 0.5) / 0.17;
                                    r = 255 * t;
                                    g = 255;
                                    b = 0;
                                } else if (hue < 0.84) {
                                    const t = (hue - 0.67) / 0.17;
                                    r = 255;
                                    g = 255 * (1 - t * 0.5);
                                    b = 0;
                                } else {
                                    const t = (hue - 0.84) / 0.16;
                                    r = 255;
                                    g = 128 * (1 - t);
                                    b = 0;
                                }

                                data[idx] = Math.min(255, currentR + glowStrength * r);
                                data[idx + 1] = Math.min(255, currentG + glowStrength * g);
                                data[idx + 2] = Math.min(255, currentB + glowStrength * b);
                            }
                        }
                    }
                }
            });
        });
    }

    // Put the modified image data back to canvas
    ctx.putImageData(imageData, 0, 0);
}

// ============================================================================
// FACE DETECTION
// ============================================================================

/**
 * Initialize face detector
 */
async function initFaceDetector() {
    console.log('DEBUG: Starting face detector initialization...');
    try {
        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        const detectorConfig = {
            runtime: 'tfjs',
            refineLandmarks: false,  // Faster without eye/iris refinement
            maxFaces: 1              // Only detect one face for performance
        };
        console.log('DEBUG: Creating detector with config:', detectorConfig);
        faceDetector = await faceLandmarksDetection.createDetector(model, detectorConfig);
        console.log('✓ Face detector initialized successfully!');
    } catch (error) {
        console.error('ERROR initializing face detector:', error);
    }
}

/**
 * Detect faces in current video frame
 */
async function detectFaces() {
    if (!faceDetector) {
        console.log('DEBUG: faceDetector not initialized yet');
        return;
    }

    if (video.readyState !== 4) {
        console.log('DEBUG: Video not ready. readyState =', video.readyState, '(need 4)');
        return;
    }

    console.log('DEBUG: Running face detection...');
    console.log('  Video dimensions:', video.videoWidth, 'x', video.videoHeight);
    console.log('  Canvas dimensions:', canvas.width, 'x', canvas.height);
    console.log('  Video element visible?', video.offsetWidth > 0, video.offsetHeight > 0);

    try {
        // Try using canvas instead of video element
        // The canvas has the current frame already drawn to it (already mirrored)
        // So we don't need to flip it again
        console.log('DEBUG: Calling estimateFaces with canvas...');
        const faces = await faceDetector.estimateFaces(canvas, {flipHorizontal: false});
        console.log('DEBUG: estimateFaces returned', faces?.length || 0, 'face(s)');

        if (faces && faces.length > 0) {
            previousFaceLandmarks = currentFaceLandmarks;
            currentFaceLandmarks = faces[0].keypoints;

            // Calculate and store eye positions for trail effect
            const leftEyeInner = currentFaceLandmarks[133];
            const leftEyeOuter = currentFaceLandmarks[33];
            const rightEyeInner = currentFaceLandmarks[362];
            const rightEyeOuter = currentFaceLandmarks[263];

            const leftEye = {
                x: (leftEyeInner.x + leftEyeOuter.x) / 2,
                y: (leftEyeInner.y + leftEyeOuter.y) / 2
            };
            const rightEye = {
                x: (rightEyeInner.x + rightEyeOuter.x) / 2,
                y: (rightEyeInner.y + rightEyeOuter.y) / 2
            };

            // Add to history
            eyePositionHistory.push({ left: leftEye, right: rightEye, timestamp: performance.now() });

            // Keep only last N positions
            if (eyePositionHistory.length > MAX_EYE_HISTORY) {
                eyePositionHistory.shift();
            }

            console.log('✓ Face detected with', currentFaceLandmarks.length, 'landmarks');
            console.log('  Eye history length:', eyePositionHistory.length);
        } else {
            previousFaceLandmarks = currentFaceLandmarks;
            currentFaceLandmarks = null;
            console.log('✗ No face detected in frame');
        }
    } catch (error) {
        console.error('ERROR detecting faces:', error);
    }
}

// ============================================================================
// MODULAR FILTER SYSTEM
// ============================================================================

/**
 * Helper: Apply convolution kernel to image for edge detection
 */
function applyConvolution(imageData, kernel) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const output = new Uint8ClampedArray(data.length);

    const kernelSize = Math.sqrt(kernel.length);
    const half = Math.floor(kernelSize / 2);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let sumR = 0, sumG = 0, sumB = 0;

            // Apply kernel
            for (let ky = 0; ky < kernelSize; ky++) {
                for (let kx = 0; kx < kernelSize; kx++) {
                    const px = x + kx - half;
                    const py = y + ky - half;

                    // Handle boundaries
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        const idx = (py * width + px) * 4;
                        const k = kernel[ky * kernelSize + kx];
                        sumR += data[idx] * k;
                        sumG += data[idx + 1] * k;
                        sumB += data[idx + 2] * k;
                    }
                }
            }

            const idx = (y * width + x) * 4;
            output[idx] = Math.abs(sumR);
            output[idx + 1] = Math.abs(sumG);
            output[idx + 2] = Math.abs(sumB);
            output[idx + 3] = data[idx + 3]; // Preserve alpha
        }
    }

    // Copy output back to imageData
    for (let i = 0; i < data.length; i++) {
        data[i] = output[i];
    }

    return imageData;
}

/**
 * Helper: Create face mask from landmarks
 * Returns a Uint8Array where 255 = inside face, 0 = outside face
 */
function createFaceMask(width, height, landmarks) {
    const mask = new Uint8Array(width * height);

    if (!landmarks || landmarks.length === 0) {
        return mask; // Empty mask
    }

    // Get face contour points (MediaPipe FaceMesh face oval indices)
    // Face oval: indices 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    //            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    const faceOvalIndices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ];

    const contourPoints = faceOvalIndices.map(i => landmarks[i]).filter(p => p);

    if (contourPoints.length === 0) return mask;

    // Simple rasterization: check each pixel if it's inside the face polygon
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (isPointInPolygon(x, y, contourPoints)) {
                mask[y * width + x] = 255;
            }
        }
    }

    return mask;
}

/**
 * Helper: Check if point is inside polygon (ray casting algorithm)
 */
function isPointInPolygon(x, y, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i].x, yi = polygon[i].y;
        const xj = polygon[j].x, yj = polygon[j].y;

        const intersect = ((yi > y) !== (yj > y))
            && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

/**
 * Helper: Add random jitter/distortion to image
 */
function addJitter(imageData, intensity = 2) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const output = new Uint8ClampedArray(data.length);

    // Copy original data
    output.set(data);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            // Random offset
            const offsetX = Math.floor((Math.random() - 0.5) * intensity * 2);
            const offsetY = Math.floor((Math.random() - 0.5) * intensity * 2);

            const newX = Math.max(0, Math.min(width - 1, x + offsetX));
            const newY = Math.max(0, Math.min(height - 1, y + offsetY));

            const srcIdx = (y * width + x) * 4;
            const dstIdx = (newY * width + newX) * 4;

            output[dstIdx] = data[srcIdx];
            output[dstIdx + 1] = data[srcIdx + 1];
            output[dstIdx + 2] = data[srcIdx + 2];
            output[dstIdx + 3] = data[srcIdx + 3];
        }
    }

    // Copy output back
    for (let i = 0; i < data.length; i++) {
        data[i] = output[i];
    }

    return imageData;
}

/**
 * Filters object - each filter is a pure function that processes imageData
 * Add new filters here to extend functionality
 */
const Filters = {
    /**
     * No filter - pass through original image
     */
    none: (imageData) => {
        // No processing needed
        return imageData;
    },

    /**
     * Grayscale filter - converts to black and white
     */
    grayscale: (imageData) => {
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];

            // Standard grayscale conversion using luminosity method
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;

            data[i] = gray;     // Red
            data[i + 1] = gray; // Green
            data[i + 2] = gray; // Blue
            // data[i + 3] is alpha, leave unchanged
        }
        return imageData;
    },

    /**
     * Invert filter - creates a negative effect
     */
    invert: (imageData) => {
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
            data[i] = 255 - data[i];         // Red
            data[i + 1] = 255 - data[i + 1]; // Green
            data[i + 2] = 255 - data[i + 2]; // Blue
            // data[i + 3] is alpha, leave unchanged
        }
        return imageData;
    },

    /**
     * Spooky filter - grayscale + invert for ghostly effect
     */
    spooky: (imageData) => {
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];

            // First: Convert to grayscale
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;

            // Second: Invert the grayscale value
            const inverted = 255 - gray;

            data[i] = inverted;     // Red
            data[i + 1] = inverted; // Green
            data[i + 2] = inverted; // Blue
            // data[i + 3] is alpha, leave unchanged
        }
        return imageData;
    },

    /**
     * Haunted filter - spooky base + glowing jittery edges with color drift
     * Combines grayscale invert with edge detection and eerie glow
     */
    haunted: (imageData) => {
        const width = imageData.width;
        const height = imageData.height;

        // Calculate time-based color drift (slow oscillation)
        const elapsed = (performance.now() - startTime) / 1000; // Convert to seconds

        // Multiple sine waves at different frequencies for organic drift
        const redDrift = Math.sin(elapsed * 0.3) * 40;           // Slow red shift
        const greenDrift = Math.sin(elapsed * 0.23 + 1.5) * 50;  // Medium green shift, phase offset
        const blueDrift = Math.sin(elapsed * 0.17 + 3.0) * 60;   // Slower blue shift, different phase

        // Step 1: Create spooky base
        const baseData = new ImageData(
            new Uint8ClampedArray(imageData.data),
            width,
            height
        );
        Filters.spooky(baseData);

        // Step 2: Detect edges using Laplace kernel
        const edgeData = new ImageData(
            new Uint8ClampedArray(imageData.data),
            width,
            height
        );

        // Laplace kernel for edge detection (8-connected)
        const laplaceKernel = [
            1,  1,  1,
            1, -8,  1,
            1,  1,  1
        ];

        applyConvolution(edgeData, laplaceKernel);

        // Step 3: Add jitter to edges
        addJitter(edgeData, 3);

        // Step 4: Combine base + glowing edges with color drift
        const base = baseData.data;
        const edges = edgeData.data;
        const result = imageData.data;

        for (let i = 0; i < result.length; i += 4) {
            // Get edge intensity (use max of RGB)
            const edgeIntensity = Math.max(edges[i], edges[i + 1], edges[i + 2]);

            // Threshold edges (only show strong edges)
            const threshold = 30;

            if (edgeIntensity > threshold) {
                // Apply eerie cyan/green glow to edges with color drift
                const glowStrength = Math.min(edgeIntensity / 255, 1.0);
                result[i] = Math.min(255, Math.max(0, base[i] + glowStrength * 50 + redDrift));        // R
                result[i + 1] = Math.min(255, Math.max(0, base[i + 1] + glowStrength * 255 + greenDrift)); // G
                result[i + 2] = Math.min(255, Math.max(0, base[i + 2] + glowStrength * 200 + blueDrift));  // B
            } else {
                // Use spooky base with subtle drift
                result[i] = Math.min(255, Math.max(0, base[i] + redDrift * 0.3));
                result[i + 1] = Math.min(255, Math.max(0, base[i + 1] + greenDrift * 0.3));
                result[i + 2] = Math.min(255, Math.max(0, base[i + 2] + blueDrift * 0.3));
            }
            result[i + 3] = 255; // Full opacity
        }

        return imageData;
    },

    /**
     * Zombify filter - undead transformation with decay
     * Combines desaturation, green tint, contrast boost, and decay texture
     */
    zombify: (imageData) => {
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;

        // Time-based pulse for unsettling "blood flow" effect
        const elapsed = (performance.now() - startTime) / 1000;
        const pulse = Math.sin(elapsed * 0.8) * 0.15 + 0.85; // Oscillates between 0.7 and 1.0

        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];

            // Calculate luminosity
            const lum = 0.299 * r + 0.587 * g + 0.114 * b;

            // Step 1: Desaturate (reduce color, keep some for sickly tone)
            const desaturation = 0.6; // 60% desaturated
            let newR = lum + (r - lum) * desaturation;
            let newG = lum + (g - lum) * desaturation;
            let newB = lum + (b - lum) * desaturation;

            // Step 2: Apply sickly green/yellow tint
            newR = newR * 0.85;           // Reduce red (pale/dead skin)
            newG = newG * 1.15 + 20;      // Boost green (sickly tone)
            newB = newB * 0.70;           // Reduce blue (yellowish)

            // Step 3: Increase contrast (gaunt, sunken features)
            // Push darks darker, lights lighter
            const contrast = 1.3;
            const midpoint = 127.5;
            newR = ((newR - midpoint) * contrast + midpoint);
            newG = ((newG - midpoint) * contrast + midpoint);
            newB = ((newB - midpoint) * contrast + midpoint);

            // Step 4: Add decay texture (noise)
            // Use pixel position as pseudo-random seed for consistent texture
            const x = (i / 4) % width;
            const y = Math.floor((i / 4) / width);
            const noise = (Math.sin(x * 0.1 + y * 0.13) * Math.cos(x * 0.07 - y * 0.11)) * 15;

            // Step 5: Apply pulse effect
            newR = (newR + noise) * pulse;
            newG = (newG + noise) * pulse;
            newB = (newB + noise) * pulse;

            // Clamp values
            data[i] = Math.min(255, Math.max(0, newR));
            data[i + 1] = Math.min(255, Math.max(0, newG));
            data[i + 2] = Math.min(255, Math.max(0, newB));
            // data[i + 3] is alpha, leave unchanged
        }

        return imageData;
    },

    /**
     * Ghost Head filter - Ethereal ghost with swirly trailing eyes + spectral echo edges
     * Grayscale + inversion + displaced colored edges + rainbow eye trails
     */
    ghostHead: (imageData) => {
        const width = imageData.width;
        const height = imageData.height;
        const data = imageData.data;

        // Step 1: Detect edges on original image (before grayscale)
        const edgeData = new ImageData(
            new Uint8ClampedArray(imageData.data),
            width,
            height
        );

        // Laplace kernel for edge detection
        const laplaceKernel = [
            1,  1,  1,
            1, -8,  1,
            1,  1,  1
        ];

        applyConvolution(edgeData, laplaceKernel);
        const edges = edgeData.data;

        // Step 2: Convert base image to grayscale and invert (ghostly base)
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];

            // Convert to grayscale
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;

            // Invert for ghostly effect
            const inverted = 255 - gray;

            data[i] = inverted;
            data[i + 1] = inverted;
            data[i + 2] = inverted;
        }

        // Step 3: Draw spectral echo - displaced colored edges
        const elapsed = (performance.now() - startTime) / 1000;

        // Time-varying displacement
        const offsetX = Math.sin(elapsed * 0.5) * 8; // Oscillates -8 to +8 pixels
        const offsetY = Math.cos(elapsed * 0.3) * 8;

        // Time-varying color (cycling through spectrum SLOWLY)
        const hueShift = (elapsed * 0.05) % 1.0; // 0 to 1, cycles every 20 seconds (much slower)
        let echoR, echoG, echoB;

        // Convert hue to RGB (simple HSV to RGB for hue only, max saturation and value)
        if (hueShift < 0.17) { // Purple to Blue
            const t = hueShift / 0.17;
            echoR = 255 * (1 - t);
            echoG = 0;
            echoB = 255;
        } else if (hueShift < 0.34) { // Blue to Cyan
            const t = (hueShift - 0.17) / 0.17;
            echoR = 0;
            echoG = 255 * t;
            echoB = 255;
        } else if (hueShift < 0.5) { // Cyan to Green
            const t = (hueShift - 0.34) / 0.16;
            echoR = 0;
            echoG = 255;
            echoB = 255 * (1 - t);
        } else if (hueShift < 0.67) { // Green to Yellow
            const t = (hueShift - 0.5) / 0.17;
            echoR = 255 * t;
            echoG = 255;
            echoB = 0;
        } else if (hueShift < 0.84) { // Yellow to Orange
            const t = (hueShift - 0.67) / 0.17;
            echoR = 255;
            echoG = 255 * (1 - t * 0.5);
            echoB = 0;
        } else { // Orange to Red to Purple
            const t = (hueShift - 0.84) / 0.16;
            echoR = 255;
            echoG = 128 * (1 - t);
            echoB = 255 * t;
        }

        // Draw displaced edges with color - THICKER edges
        const threshold = 30;
        const edgeThickness = 3; // Draw edges in a 3-pixel radius for thickness

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const edgeIntensity = Math.max(edges[idx], edges[idx + 1], edges[idx + 2]);

                if (edgeIntensity > threshold) {
                    // Calculate displaced position
                    const centerX = Math.floor(x + offsetX);
                    const centerY = Math.floor(y + offsetY);

                    // Draw in a radius to make edges thicker
                    for (let dy = -edgeThickness; dy <= edgeThickness; dy++) {
                        for (let dx = -edgeThickness; dx <= edgeThickness; dx++) {
                            const dist = Math.sqrt(dx * dx + dy * dy);
                            if (dist <= edgeThickness) {
                                const newX = centerX + dx;
                                const newY = centerY + dy;

                                // Check bounds
                                if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                                    const newIdx = (newY * width + newX) * 4;
                                    const intensity = (edgeIntensity / 255) * (1 - dist / edgeThickness); // Fade with distance

                                    // Blend the colored edge onto the grayscale image
                                    data[newIdx] = Math.min(255, data[newIdx] + echoR * intensity * 0.8);
                                    data[newIdx + 1] = Math.min(255, data[newIdx + 1] + echoG * intensity * 0.8);
                                    data[newIdx + 2] = Math.min(255, data[newIdx + 2] + echoB * intensity * 0.8);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 4: Draw trailing swirly rainbow eyes from history
        if (eyePositionHistory.length > 0) {
            const currentTime = performance.now();

            // Draw each historical position with decreasing size and opacity
            eyePositionHistory.forEach((historyItem, index) => {
                const age = (currentTime - historyItem.timestamp) / 1000; // seconds
                const ageRatio = index / eyePositionHistory.length; // 0 (oldest) to 1 (newest)

                // Fade out older positions
                const opacity = ageRatio;
                const radius = 25 + (10 * ageRatio); // Bigger glow for recent positions

                // Add slight spiral/offset for swirl effect
                const spiralOffset = (1 - ageRatio) * 10; // Older positions spiral out
                const spiralAngle = ageRatio * Math.PI * 2; // One full rotation through history

                [historyItem.left, historyItem.right].forEach(eye => {
                    // Apply spiral offset
                    const eyeX = eye.x + Math.cos(spiralAngle) * spiralOffset;
                    const eyeY = eye.y + Math.sin(spiralAngle) * spiralOffset;

                    for (let dy = -radius; dy <= radius; dy++) {
                        for (let dx = -radius; dx <= radius; dx++) {
                            const dist = Math.sqrt(dx * dx + dy * dy);
                            if (dist <= radius) {
                                const px = Math.floor(eyeX + dx);
                                const py = Math.floor(eyeY + dy);
                                if (px >= 0 && px < width && py >= 0 && py < height) {
                                    const idx = (py * width + px) * 4;
                                    const glowStrength = Math.pow(1 - (dist / radius), 2) * opacity;

                                    // Rainbow swirl - full spectrum through the trail
                                    const currentR = data[idx];
                                    const currentG = data[idx + 1];
                                    const currentB = data[idx + 2];

                                    // Create rainbow: Red -> Orange -> Yellow -> Green -> Cyan -> Blue -> Purple
                                    const hue = ageRatio; // 0 (oldest) to 1 (newest)
                                    let r, g, b;

                                    if (hue < 0.17) { // Purple to Blue
                                        const t = hue / 0.17;
                                        r = 255 * (1 - t) + 0 * t;
                                        g = 0;
                                        b = 255;
                                    } else if (hue < 0.34) { // Blue to Cyan
                                        const t = (hue - 0.17) / 0.17;
                                        r = 0;
                                        g = 255 * t;
                                        b = 255;
                                    } else if (hue < 0.5) { // Cyan to Green
                                        const t = (hue - 0.34) / 0.16;
                                        r = 0;
                                        g = 255;
                                        b = 255 * (1 - t);
                                    } else if (hue < 0.67) { // Green to Yellow
                                        const t = (hue - 0.5) / 0.17;
                                        r = 255 * t;
                                        g = 255;
                                        b = 0;
                                    } else if (hue < 0.84) { // Yellow to Orange
                                        const t = (hue - 0.67) / 0.17;
                                        r = 255;
                                        g = 255 * (1 - t * 0.5);
                                        b = 0;
                                    } else { // Orange to Red
                                        const t = (hue - 0.84) / 0.16;
                                        r = 255;
                                        g = 128 * (1 - t);
                                        b = 0;
                                    }

                                    data[idx] = Math.min(255, currentR + glowStrength * r);
                                    data[idx + 1] = Math.min(255, currentG + glowStrength * g);
                                    data[idx + 2] = Math.min(255, currentB + glowStrength * b);
                                }
                            }
                        }
                    }
                });
            });
        }

        return imageData;
    },

    /**
     * Face Debug filter - visualize face detection
     * Shows detected face with green overlay and red dots for landmarks
     */
    faceDebug: (imageData) => {
        // If no face detected, just show message
        if (!currentFaceLandmarks) {
            // Draw "NO FACE DETECTED" text on canvas
            return imageData;
        }

        const width = imageData.width;
        const height = imageData.height;
        const data = imageData.data;

        // Create face mask
        const faceMask = createFaceMask(width, height, currentFaceLandmarks);

        // Overlay green tint on detected face area
        for (let i = 0; i < data.length; i += 4) {
            const pixelIndex = i / 4;
            const inFace = faceMask[pixelIndex] > 0;

            if (inFace) {
                // Add semi-transparent green overlay
                data[i] = data[i] * 0.7;          // Dim red
                data[i + 1] = Math.min(255, data[i + 1] * 0.7 + 100); // Add green
                data[i + 2] = data[i + 2] * 0.7;  // Dim blue
            }
        }

        // Draw landmark dots (every 10th landmark to avoid clutter)
        for (let i = 0; i < currentFaceLandmarks.length; i += 10) {
            const landmark = currentFaceLandmarks[i];
            const lx = Math.floor(landmark.x);
            const ly = Math.floor(landmark.y);

            // Draw a small red dot (3x3 pixels)
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const px = lx + dx;
                    const py = ly + dy;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        const idx = (py * width + px) * 4;
                        data[idx] = 255;     // Red
                        data[idx + 1] = 0;   // No green
                        data[idx + 2] = 0;   // No blue
                    }
                }
            }
        }

        // Draw eye positions with yellow circles
        const leftEye = currentFaceLandmarks[33];
        const rightEye = currentFaceLandmarks[263];

        if (leftEye && rightEye) {
            [leftEye, rightEye].forEach(eye => {
                const radius = 10;
                for (let y = -radius; y <= radius; y++) {
                    for (let x = -radius; x <= radius; x++) {
                        if (x * x + y * y <= radius * radius) {
                            const px = Math.floor(eye.x + x);
                            const py = Math.floor(eye.y + y);
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                const idx = (py * width + px) * 4;
                                data[idx] = 255;     // Red
                                data[idx + 1] = 255; // Green (= yellow)
                                data[idx + 2] = 0;   // No blue
                            }
                        }
                    }
                }
            });
        }

        return imageData;
    }
};

/**
 * Render loop - draws video frames to canvas and applies current filter
 */
function renderFrame() {
    if (!stream) return;

    // Set canvas size to match video
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    // Draw video frame to canvas (flipped horizontally for mirror effect)
    // Do this FIRST so we have a clean frame for face detection
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    ctx.restore();

    // Detect faces every 4 frames for best performance
    // The trail effect makes slower detection look intentional and fun!
    // Do this AFTER drawing the clean video frame, BEFORE applying filter
    frameCount++;
    if (frameCount % 4 === 0 && faceDetector) {
        console.log(`DEBUG: Frame ${frameCount} - triggering face detection`);
        detectFaces();
    }

    // Log status every 60 frames (~ once per second at 60fps)
    if (frameCount % 60 === 0) {
        console.log('DEBUG: Render status -', {
            frameCount,
            faceDetector: !!faceDetector,
            currentFaceLandmarks: !!currentFaceLandmarks,
            currentFilter,
            videoReady: video.readyState === 4
        });
    }

    // Apply current filter
    if (currentFilter !== 'none' && Filters[currentFilter]) {
        // Use CPU version for all filters
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const filteredData = Filters[currentFilter](imageData);
        ctx.putImageData(filteredData, 0, 0);
    }

    // Continue rendering
    animationId = requestAnimationFrame(renderFrame);
}

// ============================================================================
// WEBCAM CONTROL
// ============================================================================

// Request webcam access and display stream
async function startWebcam() {
    try {
        // Request video stream from user's webcam
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        // Attach stream to video element
        video.srcObject = stream;

        // iOS Safari compatibility: ensure muted and playsinline
        video.muted = true;
        video.playsInline = true;

        // Wait for video to be ready, then start rendering
        video.onloadedmetadata = async () => {
            try {
                // iOS Safari requires explicit play() call
                await video.play();
                startTime = performance.now();

                // Initialize face detector
                if (!faceDetector) {
                    await initFaceDetector();
                }

                // Initialize WebGL for GPU-accelerated filters
                if (!gl) {
                    initWebGL();
                }

                renderFrame();
            } catch (playError) {
                console.error('Error playing video:', playError);
                status.textContent = 'Error starting video playback.';
            }
        };

        // Update UI
        status.textContent = 'Camera active!';
        startBtn.textContent = 'Stop Camera';
        startBtn.onclick = stopWebcam;

    } catch (error) {
        console.error('Error accessing webcam:', error);

        // Provide helpful error messages
        if (error.name === 'NotAllowedError') {
            status.textContent = 'Camera access denied. Please allow camera permissions.';
        } else if (error.name === 'NotFoundError') {
            status.textContent = 'No camera found on this device.';
        } else if (error.name === 'NotReadableError') {
            status.textContent = 'Camera is already in use by another application.';
        } else {
            status.textContent = `Error: ${error.message}`;
        }
    }
}

// Stop webcam stream
function stopWebcam() {
    if (stream) {
        // Stop all tracks in the stream
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;

        // Stop render loop
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Update UI
        status.textContent = 'Camera stopped. Click "Start Camera" to begin.';
        startBtn.textContent = 'Start Camera';
        startBtn.onclick = startWebcam;
    }
}

// Set up button click handler
startBtn.onclick = startWebcam;

// ============================================================================
// FILTER CONTROLS
// ============================================================================

/**
 * Set up filter button event listeners
 */
filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Update current filter
        currentFilter = btn.dataset.filter;

        // Update button states
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

// Set initial active state
document.querySelector('[data-filter="none"]').classList.add('active');

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
