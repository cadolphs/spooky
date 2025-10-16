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
     * Ghost Head filter - Ethereal ghost with swirly trailing eyes
     * Grayscale + edge glow + rainbow eye trails
     */
    ghostHead: (imageData) => {
        const width = imageData.width;
        const height = imageData.height;
        const data = imageData.data;

        // Step 1: Detect edges using Laplace kernel (like Haunted filter)
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

        // Step 2: Convert to grayscale and invert (like Spooky filter)
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

        // Step 3: Add glowing edge overlay on top
        for (let i = 0; i < data.length; i += 4) {
            // Get edge intensity (use max of RGB from edge detection)
            const edgeIntensity = Math.max(edges[i], edges[i + 1], edges[i + 2]);

            // Lower threshold to show more edges
            const threshold = 15;

            if (edgeIntensity > threshold) {
                // Add bright cyan/white glow to edges
                const glowStrength = Math.min(edgeIntensity / 255, 1.0);
                data[i] = Math.min(255, data[i] + glowStrength * 150);       // R
                data[i + 1] = Math.min(255, data[i + 1] + glowStrength * 200); // G
                data[i + 2] = Math.min(255, data[i + 2] + glowStrength * 255); // B
            }
        }

        // Draw trailing swirly eyes from history
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
