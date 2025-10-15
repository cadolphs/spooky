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
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    ctx.restore();

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

        // Wait for video to be ready, then start rendering
        video.onloadedmetadata = () => {
            startTime = performance.now();
            renderFrame();
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
