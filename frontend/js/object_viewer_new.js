// New Object Viewer - Local Crop View
function updateObjectDisplay() {
    const objects = objectsData[currentObjectType];
    const display = document.getElementById('objectDisplay');
    const info = document.getElementById('objectInfo');
    const prevBtn = document.getElementById('prevObjectBtn');
    const nextBtn = document.getElementById('nextObjectBtn');

    if (!display || !info || !prevBtn || !nextBtn) {
        console.warn('⚠️ Object view elements not found');
        return;
    }

    console.log(`🔍 Update object display: ${currentObjectType}, index: ${currentObjectIndex}, total: ${objects.length}`);

    // Update object information
    info.textContent = `${currentObjectIndex + 1} / ${objects.length}`;

    // Update button states
    prevBtn.disabled = currentObjectIndex === 0 || objects.length === 0;
    nextBtn.disabled = currentObjectIndex === objects.length - 1 || objects.length === 0;

    prevBtn.style.opacity = prevBtn.disabled ? '0.5' : '1';
    nextBtn.style.opacity = nextBtn.disabled ? '0.5' : '1';

    // Display object local view
    if (objects.length > 0 && objects[currentObjectIndex]) {
        const currentObject = objects[currentObjectIndex];
        console.log('🖼️ Display object local view:', currentObject);

        // Build local view image URLs
        const runNumber = String(window.currentRunNumber || '042').padStart(3, '0');
        const baseURL = `${API_BASE_URL}/files/results/runs/run_${runNumber}/detection`;

        const beforeImageUrl = `${baseURL}/${currentObject.before_path}`;
        const afterImageUrl = `${baseURL}/${currentObject.after_path}`;
        const maskImageUrl = `${baseURL}/${currentObject.mask_path}`;

        console.log('🖼️ Local view URLs:');
        console.log('  Before:', beforeImageUrl);
        console.log('  After:', afterImageUrl);
        console.log('  Mask:', maskImageUrl);

        display.innerHTML = `
            <div style="display: flex; width: 100%; height: calc(100% - 120px); gap: 10px;">
                <!-- Before Change Image -->
                <div style="width: 50%; height: 100%; position: relative; border: 2px solid #ddd; border-radius: 8px; overflow: hidden;">
                    <img id="beforeImage" src="${beforeImageUrl}"
                         style="width: 100%; height: 100%; object-fit: contain; background: #f8f9fa;"
                         alt="Before Change - Local Area"
                         onload="console.log('✅ Before local image loaded successfully')"
                         onerror="console.error('❌ Before local image loading failed:', this.src)">

                    <!-- Before Mask Overlay -->
                    <img id="beforeMaskOverlay" src="${maskImageUrl}"
                         style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; display: none; pointer-events: none;"
                         alt="Before Mask Overlay"
                         onload="console.log('✅ Before Mask loaded successfully')"
                         onerror="console.error('❌ Before Mask loading failed:', this.src)">

                    <!-- Label -->
                    <div style="position: absolute; top: 8px; left: 8px; background: rgba(255,0,0,0.8); color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                        Before Change
                    </div>
                </div>

                <!-- After Change Image -->
                <div style="width: 50%; height: 100%; position: relative; border: 2px solid #ddd; border-radius: 8px; overflow: hidden;">
                    <img id="afterImage" src="${afterImageUrl}"
                         style="width: 100%; height: 100%; object-fit: contain; background: #f8f9fa;"
                         alt="After Change - Local Area"
                         onload="console.log('✅ After local image loaded successfully')"
                         onerror="console.error('❌ After local image loading failed:', this.src)">

                    <!-- After Mask Overlay -->
                    <img id="afterMaskOverlay" src="${maskImageUrl}"
                         style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; display: none; pointer-events: none;"
                         alt="After Mask Overlay"
                         onload="console.log('✅ After Mask loaded successfully')"
                         onerror="console.error('❌ After Mask loading failed:', this.src)">

                    <!-- Label -->
                    <div style="position: absolute; top: 8px; left: 8px; background: rgba(0,128,0,0.8); color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                        After Change
                    </div>
                </div>
            </div>

            <!-- Control Area -->
            <div style="margin-top: 15px; padding: 12px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;">
                <!-- Mask Control -->
                <div style="margin-bottom: 10px;">
                    <label style="display: inline-flex; align-items: center; gap: 8px; cursor: pointer; font-size: 14px; font-weight: 500;">
                        <input type="checkbox" id="maskToggle" onchange="toggleMaskDisplay()"
                               style="transform: scale(1.3); cursor: pointer;">
                        <span>🎭 Display Mask Overlay</span>
                    </label>
                </div>

                <!-- Object Information -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 13px;">
                    <div>
                        <strong style="color: #495057;">${translateObjectName ? translateObjectName(currentObject.name) : currentObject.name}</strong><br>
                        <span style="color: #6c757d;">Confidence: ${currentObject.confidence}%</span>
                    </div>
                    <div>
                        <span style="color: #6c757d;">Area Size: ${currentObject.bbox ? `${currentObject.bbox.width}×${currentObject.bbox.height}px` : 'Unknown'}</span><br>
                        <span style="color: #6c757d;">Position: ${currentObject.bbox ? `(${currentObject.bbox.x}, ${currentObject.bbox.y})` : 'Unknown'}</span>
                    </div>
                </div>
            </div>
        `;
    } else {
        display.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #6c757d; flex-direction: column; text-align: center;">
                <div style="font-size: 64px; margin-bottom: 20px; opacity: 0.5;">📷</div>
                <div style="font-size: 18px; margin-bottom: 8px; font-weight: 500;">No ${currentObjectType === 'disappeared' ? 'disappeared' : 'new'} objects</div>
                <div style="font-size: 14px; opacity: 0.7;">Please run detection to generate object change results</div>
            </div>
        `;
    }
}

// Toggle mask display function
function toggleMaskDisplay() {
    const beforeMaskOverlay = document.getElementById('beforeMaskOverlay');
    const afterMaskOverlay = document.getElementById('afterMaskOverlay');
    const maskToggle = document.getElementById('maskToggle');

    if (!beforeMaskOverlay || !afterMaskOverlay || !maskToggle) {
        console.warn('⚠️ Mask elements not found');
        return;
    }

    const isChecked = maskToggle.checked;

    if (isChecked) {
        beforeMaskOverlay.style.display = 'block';
        afterMaskOverlay.style.display = 'block';
        console.log('✅ Mask overlay displayed');
    } else {
        beforeMaskOverlay.style.display = 'none';
        afterMaskOverlay.style.display = 'none';
        console.log('❌ Mask overlay hidden');
    }
}
