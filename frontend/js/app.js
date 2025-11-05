// ===== Sidebar Toggle Function =====
function switchView(view) {
    // Update sidebar icon status
    document.getElementById('icon-recognition').classList.toggle('active', view === 'recognition');
    document.getElementById('icon-history').classList.toggle('active', view === 'history');

    // Switch main content area display
    document.getElementById('recognition-area').style.display = (view === 'recognition') ? 'block' : 'none';
    document.getElementById('history-area').style.display = (view === 'history') ? 'block' : 'none';

    // If switching to history review, load history data
    if (view === 'history') {
        loadHistoryData();
    } else if (view === 'recognition') {
        // 🔧 When switching back to normal mode, restore original object display function
        restoreOriginalObjectDisplay();
    }
}

// ===== Load History Review Data =====
async function loadHistoryData() {
    const container = document.getElementById('historyThumbnails');
    container.innerHTML = '<div style="text-align:center;color:white;padding:20px;">Loading...</div>';

    try {
        const response = await fetch(`${API_BASE_URL}/history/runs`);
        const data = await response.json();

        if (!data.runs || data.runs.length === 0) {
            container.innerHTML = '<div style="text-align:center;color:white;padding:40px;font-size:18px;">No historical analysis records</div>';
            return;
        }

        // Render history thumbnails
        container.innerHTML = data.runs.map(run => {
            console.log('Rendering run:', run); // For debugging

            let imageContent = '';

            if (run.image1_url || run.image2_url) {
                // Fix: Add complete server URL
                const image1Url = run.image1_url ? `${API_BASE_URL.replace('/api', '')}${run.image1_url}` : null;
                const image2Url = run.image2_url ? `${API_BASE_URL.replace('/api', '')}${run.image2_url}` : null;

                imageContent = `
                    <div class="image-container">
                        ${image1Url ? `<img src="${image1Url}" alt="Image 1" class="image1" onerror="console.log('Image1 loading failed:', this.src)">` : ''}
                        ${image2Url ? `<img src="${image2Url}" alt="Image 2" class="image2" onerror="console.log('Image2 loading failed:', this.src)">` : ''}
                    </div>
                `;
            } else {
                imageContent = '<div class="no-image">No Images</div>';
            }

            return `
                <div class="history-thumbnail" onclick="showHistoryDetail('${run.run_id}')">
                    ${imageContent}
                    <div class="title">${run.run_id}</div>
                </div>
            `;
        }).join('');

    } catch (error) {
        container.innerHTML = '<div style="text-align:center;color:#ff6b6b;padding:40px;font-size:16px;">Loading failed: ' + error.message + '</div>';
    }
}

// ===== Show Historical Detailed Analysis =====
async function showHistoryDetail(runId) {
    const detailContainer = document.getElementById('historyDetail');
    detailContainer.innerHTML = '<div style="text-align:center;color:white;padding:20px;">Loading analysis results...</div>';

    try {
        const response = await fetch(`${API_BASE_URL}/history/run/${runId}`);
        const data = await response.json();

        console.log('Historical analysis data:', data);

        // 🔧 Setup history viewing environment, simulate real-time analysis global variables
        setupHistoryEnvironment(data, runId);

        // 🔧 Use exactly the same interactive viewer as original analysis
        const viewerHTML = createInteractiveViewer(data);

        detailContainer.innerHTML = `
            <div class="history-detail" style="background: rgba(255,255,255,0.95); border-radius: 15px; padding: 25px;">
                <div class="history-header" style="margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 10px;">
                    <h3 style="margin: 0; font-size: 18px;">📊 Analysis Results Details - ${runId}</h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Analysis time: ${data.timestamp || 'Unknown'}</p>
                </div>
                <div class="result-item">
                    ${viewerHTML}
                </div>
            </div>
        `;

        // 🔧 Initialize exactly the same interactive functionality as original analysis
        await initializeHistoryInteractiveViewer(runId);

    } catch (error) {
        console.error('Loading history details error:', error);
        detailContainer.innerHTML = '<div style="text-align:center;color:#ff6b6b;padding:40px;font-size:16px;">Failed to load analysis results</div>';
    }
}

// 🔧 Added: Setup history viewing environment (simulate real-time analysis global variables)
function setupHistoryEnvironment(data, runId) {
    // 🔧 Setup separated image data, using actual historical file path format
    window.separatedImages = {
        // Original images (usually image1.jpg, image2.jpg in historical data)
        image1_original: `${runId}/upload/image1.jpg`,
        image2_original: `${runId}/upload/image2.jpg`,
        // Mask images (possibly in detection directory)
        image1_same_masks: `${runId}/detection/image1_same_masks.jpg`,
        image2_same_masks: `${runId}/detection/image2_same_masks.jpg`,
        image1_disappeared_masks: `${runId}/detection/image1_disappeared_masks.jpg`,
        image2_appeared_masks: `${runId}/detection/image2_appeared_masks.jpg`
    };

    // Set detection results, simulating real-time analysis detectionResults
    window.detectionResults = {
        data: data
    };

    // 🔧 Setup global variables required for object view
    window.objectsData = {
        disappeared: data.disappeared_objects || [],
        appeared: data.appeared_objects || []
    };

    // 🔧 同時設置全域變數，讓原始 updateObjectDisplay 函數能夠存取
    // 將 window.objectsData 同步到全域作用域
    window.objectsData = window.objectsData;
    window.currentObjectType = 'disappeared';
    window.currentObjectIndex = 0;

    // 直接設置全域變數
    objectsData = window.objectsData;
    currentObjectType = window.currentObjectType;
    currentObjectIndex = window.currentObjectIndex;

    console.log('✅ 全域變數已設置:', {
        objectsData: objectsData,
        currentObjectType: currentObjectType,
        currentObjectIndex: currentObjectIndex,
        disappeared_count: objectsData.disappeared.length,
        appeared_count: objectsData.appeared.length
    });

    // 🔧 確保歷史物件包含完整的統計資訊
    const processObjectsWithStats = (objects) => {
        return objects.map(obj => {
            // 確保物件包含所有必要的統計屬性
            return {
                ...obj,
                // 如果沒有 changeRatio，計算一個合理的值
                changeRatio: obj.changeRatio || obj.change_ratio || Math.round((obj.confidence || 75) * 0.9),
                // 如果沒有 confidence，使用現有值或預設值
                confidence: obj.confidence || obj.score || 85,
                // 確保有 bbox 資訊
                bbox: obj.bbox || {
                    width: obj.width || 120,
                    height: obj.height || 100,
                    x: obj.x || 0,
                    y: obj.y || 0
                },
                // 確保有名稱
                name: obj.name || (obj.class_name ? obj.class_name : 'Unknown Object')
            };
        });
    };

    // 處理消失和出現的物件
    window.objectsData.disappeared = processObjectsWithStats(window.objectsData.disappeared);
    window.objectsData.appeared = processObjectsWithStats(window.objectsData.appeared);

    // 🔧 Add debug information: Check object data content with stats
    console.log('🔍 Object data debug with stats:');
    console.log('Disappeared objects total:', window.objectsData.disappeared.length);
    console.log('New objects total:', window.objectsData.appeared.length);

    if (window.objectsData.disappeared && window.objectsData.disappeared.length > 0) {
        console.log('First disappeared object with stats:', window.objectsData.disappeared[0]);
    }

    if (window.objectsData.appeared && window.objectsData.appeared.length > 0) {
        console.log('First appeared object with stats:', window.objectsData.appeared[0]);
    }

    // 現在設置其他全域變數
    window.currentObjectType = 'disappeared';
    window.currentObjectIndex = 0;

    // Set current mask type
    window.currentMaskType = 'different'; // Default to show changes (different objects)
    window.sliderPosition = 50;
    window.masksVisible = false; // Start with masks off, user can toggle them
    window.maskOpacity = 0.7; // 設置預設遮罩透明度

    // 🔧 Set run number (extracted from runId)
    const runMatch = runId.match(/run_(\d+)/);
    if (runMatch) {
        window.currentRunNumber = parseInt(runMatch[1]);
    }

    console.log('🔧 History environment setup complete:', {
        separatedImages: window.separatedImages,
        detectionResults: window.detectionResults,
        objectsData: window.objectsData,
        currentObjectType: window.currentObjectType,
        currentRunNumber: window.currentRunNumber,
        runId: runId
    });
}

// 🔧 Added: Initialize history interactive viewer (same as original analysis)
async function initializeHistoryInteractiveViewer(runId) {
    console.log('🎮 Initialize history interactive viewer...');

    // 🔧 先覆寫圖片載入函數，使用歷史檔案路徑
    overrideImageLoadingForHistory(runId);

    // 🔧 Override object view function, use historical run number
    overrideObjectDisplayForHistory(runId);

    // 🔧 覆寫物件切換函數，確保歷史查看器兼容性
    overrideObjectFunctionsForHistory();

    // 使用與原始分析相同的初始化函數
    await initializeInteractiveViewer();

    console.log('✅ History interactive viewer initialization complete');
}

// 🔧 新增：覆寫物件檢視切換函數，確保歷史查看器兼容性
function overrideObjectFunctionsForHistory() {
    // 覆寫switchObjectType函數
    window.switchObjectType = async function(type) {
        window.currentObjectType = type;
        window.currentObjectIndex = 0;

        // 更新按鈕狀態
        document.querySelectorAll('.object-tab').forEach(btn => {
            if (btn.textContent.includes(type === 'disappeared' ? '消失' : '新增')) {
                btn.style.background = '#667eea';
                btn.style.color = 'white';
            } else {
                btn.style.background = 'transparent';
                btn.style.color = '#333';
            }
        });

        await window.updateObjectDisplay();
    };

    // 覆寫previousObject函數
    window.previousObject = async function() {
        if (window.currentObjectIndex > 0) {
            window.currentObjectIndex--;
            await window.updateObjectDisplay();
        }
    };

    // 覆寫nextObject函數
    window.nextObject = async function() {
        const objects = window.objectsData[window.currentObjectType];
        if (window.currentObjectIndex < objects.length - 1) {
            window.currentObjectIndex++;
            await window.updateObjectDisplay();
        }
    };

    console.log('🔧 物件檢視函數已覆寫為歷史兼容版本');
}

// 🔧 新增：覆寫物件檢視切換函數，確保歷史查看器兼容性
function overrideObjectFunctionsForHistory() {
    // 覆寫switchObjectType函數
    window.switchObjectType = async function(type) {
        window.currentObjectType = type;
        window.currentObjectIndex = 0;

        // 更新按鈕狀態
        document.querySelectorAll('.object-tab').forEach(btn => {
            if (btn.textContent.includes(type === 'disappeared' ? '消失' : '新增')) {
                btn.style.background = '#667eea';
                btn.style.color = 'white';
            } else {
                btn.style.background = 'transparent';
                btn.style.color = '#333';
            }
        });

        await window.updateObjectDisplay();
    };

    // 覆寫previousObject函數
    window.previousObject = async function() {
        if (window.currentObjectIndex > 0) {
            window.currentObjectIndex--;
            await window.updateObjectDisplay();
        }
    };

    // 覆寫nextObject函數
    window.nextObject = async function() {
        const objects = window.objectsData[window.currentObjectType];
        if (window.currentObjectIndex < objects.length - 1) {
            window.currentObjectIndex++;
            await window.updateObjectDisplay();
        }
    };

    // 覆寫toggleMaskDisplay函數，用於歷史物件遮罩切換
    window.toggleMaskDisplay = function() {
        const beforeMask = document.getElementById('objectBeforeMask');
        const afterMask = document.getElementById('objectAfterMask');
        const maskToggle = document.getElementById('objectMaskToggle');
        const toggleSwitch = maskToggle ? maskToggle.parentNode.querySelector('.toggle-switch') : null;
        const toggleHandle = toggleSwitch ? toggleSwitch.querySelector('.toggle-handle') : null;

        if (!maskToggle) {
            console.warn('⚠️ 歷史遮罩切換元素未找到');
            return;
        }

        const isChecked = maskToggle.checked;
        console.log('🎭 歷史遮罩切換:', isChecked);

        // 更新切換開關視覺狀態
        if (toggleSwitch && toggleHandle) {
            if (isChecked) {
                toggleSwitch.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
                toggleHandle.style.transform = 'translateX(24px)';
            } else {
                toggleSwitch.style.background = '#ddd';
                toggleHandle.style.transform = 'translateX(2px)';
            }
        }

        // 控制遮罩顯示/隱藏 - 支援 canvas 元素
        if (beforeMask) beforeMask.style.display = isChecked ? 'block' : 'none';
        if (afterMask) afterMask.style.display = isChecked ? 'block' : 'none';
    };

    console.log('🔧 物件檢視函數已覆寫為歷史兼容版本');
}

// 🔧 新增：覆寫物件顯示，使用歷史運行編號
function overrideObjectDisplayForHistory(runId) {
    // 保存原始函數的引用
    if (!window.originalUpdateObjectDisplay) {
        window.originalUpdateObjectDisplay = window.updateObjectDisplay;
    }

    if (window.originalUpdateObjectDisplay) {
        window.updateObjectDisplay = async function() {
            // 🔧 確保變數同步
            objectsData = window.objectsData;
            currentObjectType = window.currentObjectType;
            currentObjectIndex = window.currentObjectIndex;

            // 🔧 先調用原始函數來生成完整的HTML結構（包括統計卡片）
            await window.originalUpdateObjectDisplay();

            // 然後進行歷史模式特定的修改
            const objects = window.objectsData[window.currentObjectType];
            console.log(`🔧 物件顯示調試:`, {
                currentObjectType: window.currentObjectType,
                currentObjectIndex: window.currentObjectIndex,
                objectsData: window.objectsData,
                selectedObjects: objects,
                selectedObjectsLength: objects?.length || 0
            });

            if (!objects || objects.length === 0) {
                console.log('📭 沒有物件可顯示');
                return;
            }

            console.log(`🔍 更新歷史物件顯示: ${window.currentObjectType}, 索引: ${window.currentObjectIndex}, 總數: ${objects.length}`);

            const currentObject = objects[window.currentObjectIndex];
            if (!currentObject) {
                console.warn('⚠️ 當前物件不存在');
                return;
            }

            // 🔧 添加詳細調試信息
            console.log('🔍 當前顯示的物件詳情:');
            console.log('  - 物件類型 (選擇的):', window.currentObjectType);
            console.log('  - 物件索引:', window.currentObjectIndex);
            console.log('  - 物件數據:', currentObject);
            console.log('  - 物件名稱:', currentObject.name);
            console.log('  - before_path:', currentObject.before_path);
            console.log('  - after_path:', currentObject.after_path);
            console.log('  - mask_path:', currentObject.mask_path);

            console.log('🖼️ 顯示歷史物件:', currentObject);

            // 🔧 歷史模式特定：更新圖片路徑為歷史檔案路徑
            if (currentObject.before_path && currentObject.after_path) {
                let beforePath = currentObject.before_path.replace(/\\/g, '/');
                let afterPath = currentObject.after_path.replace(/\\/g, '/');

                // 🔧 處理相對路徑，構建完整的歷史檔案路徑
                if (!beforePath.startsWith(runId)) {
                    beforePath = `${runId}/detection/${beforePath}`;
                }
                if (!afterPath.startsWith(runId)) {
                    afterPath = `${runId}/detection/${afterPath}`;
                }

                console.log('🔧 構建的圖片路徑:');
                console.log('  - beforePath:', beforePath);
                console.log('  - afterPath:', afterPath);
                console.log('  - runId:', runId);

                // 更新物件圖片顯示
                const beforeImg = document.getElementById('beforeImage');
                const afterImg = document.getElementById('afterImage');

                console.log('🔧 圖片元素檢查:');
                console.log('  - beforeImg存在:', !!beforeImg);
                console.log('  - afterImg存在:', !!afterImg);

                if (beforeImg && afterImg) {
                    const beforeImgUrl = `${API_BASE_URL}/files/${beforePath}`;
                    const afterImgUrl = `${API_BASE_URL}/files/${afterPath}`;

                    beforeImg.src = beforeImgUrl;
                    afterImg.src = afterImgUrl;

                    console.log('📸 已設置歷史物件圖片URL:');
                    console.log('  - 前圖URL:', beforeImgUrl);
                    console.log('  - 後圖URL:', afterImgUrl);

                    // 🔧 添加錯誤處理，確保圖片載入成功
                    beforeImg.onload = function() {
                        console.log('✅ 前圖載入成功');
                    };
                    beforeImg.onerror = function() {
                        console.error('❌ 前圖載入失敗:', beforeImgUrl);
                    };

                    afterImg.onload = function() {
                        console.log('✅ 後圖載入成功');
                    };
                    afterImg.onerror = function() {
                        console.error('❌ 後圖載入失敗:', afterImgUrl);
                    };
                } else {
                    console.error('❌ 找不到物件圖片元素');
                }

                // 🔧 載入遮罩 - 根據物件類型使用正確的遮罩邏輯
                const beforeMaskCanvas = document.getElementById('beforeMaskCanvas');
                const afterMaskCanvas = document.getElementById('afterMaskCanvas');

                console.log('🎭 準備載入遮罩:');
                console.log('  - 物件類型:', window.currentObjectType);
                console.log('  - 物件遮罩路徑:', currentObject.mask_path);
                console.log('  - beforeMaskCanvas元素:', !!beforeMaskCanvas);
                console.log('  - afterMaskCanvas元素:', !!afterMaskCanvas);

                if (beforeMaskCanvas && afterMaskCanvas && currentObject.mask_path) {
                    // 載入歷史遮罩
                    loadHistoryColoredMask(currentObject.mask_path, window.currentObjectType, runId);
                } else {
                    console.log('⚠️ 遮罩元素或路徑缺失');
                }
            }
        };
    }
}

// 🔧 新增：恢復原始物件顯示函數
function restoreOriginalObjectDisplay() {
    if (window.originalUpdateObjectDisplay) {
        window.updateObjectDisplay = window.originalUpdateObjectDisplay;
        console.log('🔄 已恢復原始物件顯示函數');
    }
}

// 🔧 新增：歷史模式專用的彩色遮罩載入函數
function loadHistoryColoredMask(maskPath, objectType, runId) {
    console.log('🎭 載入歷史遮罩:', maskPath, '類型:', objectType, '運行ID:', runId);

    // 確定遮罩顏色
    const maskColor = objectType === 'disappeared' ?
        { r: 255, g: 0, b: 0 } :    // 紅色 - 消失
        { r: 0, g: 255, b: 0 };     // 綠色 - 新增

    // 載入並處理遮罩
    const maskImage = new Image();
    maskImage.crossOrigin = 'anonymous';

    maskImage.onload = function() {
        drawHistoryColoredMask('beforeMaskCanvas', this, maskColor);
        drawHistoryColoredMask('afterMaskCanvas', this, maskColor);

        // 顯示遮罩
        const beforeCanvas = document.getElementById('beforeMaskCanvas');
        const afterCanvas = document.getElementById('afterMaskCanvas');
        if (beforeCanvas) beforeCanvas.style.display = 'block';
        if (afterCanvas) afterCanvas.style.display = 'block';

        console.log(`✅ 載入歷史${objectType === 'disappeared' ? '紅色消失' : '綠色新增'}遮罩成功`);
    };

    maskImage.onerror = function() {
        console.error('❌ 載入歷史遮罩圖片失敗:', maskPath);
    };

    // 構建遮罩圖片URL
    let cleanMaskPath = maskPath.replace(/\\/g, '/');
    if (!cleanMaskPath.startsWith(runId)) {
        cleanMaskPath = `${runId}/detection/${maskPath}`;
    }
    const maskImageUrl = `${API_BASE_URL}/files/${cleanMaskPath}`;

    console.log('🔗 歷史遮罩圖片URL:', maskImageUrl);
    maskImage.src = maskImageUrl;
}

// 🔧 新增：歷史模式專用的遮罩繪製函數
function drawHistoryColoredMask(canvasId, maskImage, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.warn('⚠️ Canvas元素未找到:', canvasId);
        return;
    }

    const ctx = canvas.getContext('2d');
    const container = canvas.parentElement;

    // 獲取對應的圖片元素來計算正確的尺寸
    const imageId = canvasId.includes('before') ? 'beforeImage' : 'afterImage';
    const img = document.getElementById(imageId);

    if (!img) {
        console.warn('⚠️ 對應的圖片元素未找到:', imageId);
        return;
    }

    // 設置canvas的實際尺寸為容器尺寸
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    // 清除canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 等待圖片載入完成後再計算尺寸
    if (img.complete && img.naturalWidth > 0) {
        drawMaskOnCanvas();
    } else {
        img.onload = drawMaskOnCanvas;
    }

    function drawMaskOnCanvas() {
        // 計算圖片在容器中的實際顯示區域 (object-fit: contain 的效果)
        const containerAspect = container.clientWidth / container.clientHeight;
        const imageAspect = img.naturalWidth / img.naturalHeight;

        let displayWidth, displayHeight, offsetX, offsetY;

        if (containerAspect > imageAspect) {
            // 容器比圖片寬，圖片會垂直填滿，左右留白
            displayHeight = container.clientHeight;
            displayWidth = displayHeight * imageAspect;
            offsetX = (container.clientWidth - displayWidth) / 2;
            offsetY = 0;
        } else {
            // 容器比圖片高，圖片會水平填滿，上下留白
            displayWidth = container.clientWidth;
            displayHeight = displayWidth / imageAspect;
            offsetX = 0;
            offsetY = (container.clientHeight - displayHeight) / 2;
        }

        // 創建臨時canvas來處理遮罩
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = maskImage.width;
        tempCanvas.height = maskImage.height;

        // 繪製原始遮罩到臨時canvas
        tempCtx.drawImage(maskImage, 0, 0);

        // 獲取圖像數據
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;

        // 將白色區域替換為指定顏色
        for (let i = 0; i < data.length; i += 4) {
            const alpha = data[i + 3];
            if (alpha > 128) { // 如果像素不透明
                data[i] = color.r;     // 紅色分量
                data[i + 1] = color.g; // 綠色分量
                data[i + 2] = color.b; // 藍色分量
                data[i + 3] = 180;     // 透明度 (70% 不透明)
            }
        }

        // 將處理後的數據放回
        tempCtx.putImageData(imageData, 0, 0);

        // 繪製遮罩到正確的位置和尺寸
        ctx.drawImage(tempCanvas, offsetX, offsetY, displayWidth, displayHeight);

        console.log(`✅ 歷史遮罩繪製完成: ${canvasId}, 尺寸: ${displayWidth.toFixed(0)}x${displayHeight.toFixed(0)}, 偏移: ${offsetX.toFixed(0)},${offsetY.toFixed(0)}`);
    }
}

// 🔧 新增：覆寫圖片載入，使用歷史檔案路徑
function overrideImageLoadingForHistory(runId) {
    // 覆寫原始圖片載入函數
    const originalLoadOriginalImages = window.loadOriginalImages;
    window.loadOriginalImages = function() {
        const layer1 = document.getElementById('imageLayer1');
        const layer2 = document.getElementById('imageLayer2');

        if (layer1 && layer2) {
            // 🔧 使用實際的歷史檔案路徑
            const image1Path = `${runId}/upload/image1.jpg`;  // 第一張圖片
            const image2Path = `${runId}/upload/image2.jpg`;  // 第二張圖片

            // 設置背景圖片
            layer1.style.backgroundImage = `url(${API_BASE_URL}/files/${image2Path})`;  // 底層顯示第二張
            layer2.style.backgroundImage = `url(${API_BASE_URL}/files/${image1Path})`;  // 上層顯示第一張

            console.log('📸 已載入歷史拉桿圖片:');
            console.log('  - 左側（上層）:', `${API_BASE_URL}/files/${image1Path}`);
            console.log('  - 右側（底層）:', `${API_BASE_URL}/files/${image2Path}`);
        }
    };

    // 覆寫遮罩圖片載入函數
    const originalLoadImagesWithMasks = window.loadImagesWithMasks;
    if (originalLoadImagesWithMasks) {
        window.loadImagesWithMasks = function() {
            console.log('🎭 [歷史模式] loadImagesWithMasks 被調用');
            console.log('  - window.currentMaskType:', window.currentMaskType);
            console.log('  - window.masksVisible:', window.masksVisible);
            console.log('  - runId:', runId);

            // 先確保原始圖片已載入
            window.loadOriginalImages();

            // 根據當前遮罩類型載入相應的歷史遮罩圖片
            const layer1 = document.getElementById('imageLayer1');
            const layer2 = document.getElementById('imageLayer2');

            console.log('  - layer1 存在:', !!layer1);
            console.log('  - layer2 存在:', !!layer2);

            if (layer1 && layer2) {
                // 🔧 先清除所有遮罩疊加層
                removeHistoryMaskOverlays(layer1);
                removeHistoryMaskOverlays(layer2);

                // 🔧 檢查遮罩是否應該顯示
                const shouldShowMasks = window.masksVisible !== undefined ? window.masksVisible : masksVisible;

                if (shouldShowMasks && window.currentMaskType && window.currentMaskType !== 'none') {
                    let image1MaskPath, image2MaskPath;

                    switch(window.currentMaskType) {
                        case 'same':
                            image1MaskPath = `${runId}/detection/image1_same_masks.png`;
                            image2MaskPath = `${runId}/detection/image2_same_masks.png`;
                            console.log('🟡 Loading same object mask images:', image1MaskPath, image2MaskPath);
                            break;
                        case 'different':
                            image1MaskPath = `${runId}/detection/image1_disappeared_masks.png`;
                            image2MaskPath = `${runId}/detection/image2_appeared_masks.png`;
                            console.log('🔴🟢 Loading different object mask images:', image1MaskPath, image2MaskPath);
                            break;
                    }

                    if (image1MaskPath && image2MaskPath) {
                        // 為每個圖層添加遮罩疊加層
                        addHistoryMaskOverlay(layer2, `${API_BASE_URL}/files/${image1MaskPath}`, 'history-mask-layer2');
                        addHistoryMaskOverlay(layer1, `${API_BASE_URL}/files/${image2MaskPath}`, 'history-mask-layer1');

                        console.log('✅ 已添加歷史遮罩疊加層');
                    }
                } else {
                    console.log('⚪ 遮罩已關閉或遮罩類型為 none，不顯示遮罩');
                }
            }
        };
    }

    // 立即載入原始圖片
    window.loadOriginalImages();

    // 確保遮罩透明度變數被正確初始化
    if (typeof window.maskOpacity === 'undefined') {
        window.maskOpacity = 0.7;
    }
}

// 🔧 新增：歷史模式專用的遮罩疊加函數（參考正常模式）
function addHistoryMaskOverlay(targetElement, maskUrl, maskId) {
    console.log(`🎭 開始添加歷史遮罩疊加層: ${maskId}, 路徑: ${maskUrl}`);

    // 🔧 參考正常模式：先移除現有的遮罩疊加層（簡單直接）
    const existingOverlay = targetElement.querySelector('.mask-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
        console.log('🗑️ 移除現有遮罩疊加層:', existingOverlay.id);
    }

    // 同時移除歷史模式的遮罩疊加層
    const existingHistoryOverlay = targetElement.querySelector('.history-mask-overlay');
    if (existingHistoryOverlay) {
        existingHistoryOverlay.remove();
        console.log('🗑️ 移除現有歷史遮罩疊加層:', existingHistoryOverlay.id);
    }

    // 檢查目標元素是否存在
    if (!targetElement) {
        console.error('❌ 目標元素不存在:', targetElement);
        return;
    }

    // 🔧 參考正常模式：創建遮罩疊加層（使用相同的 className）
    const overlay = document.createElement('div');
    overlay.className = 'mask-overlay';  // 使用與正常模式相同的 className
    overlay.id = maskId;
    overlay.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url(${maskUrl});
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        opacity: ${window.maskOpacity || 0.7};
        pointer-events: none;
        z-index: 5;
        transition: opacity 0.2s ease;
    `;

    // 確保目標元素有相對定位
    if (getComputedStyle(targetElement).position === 'static') {
        targetElement.style.position = 'relative';
    }

    // 添加到目標元素
    targetElement.appendChild(overlay);

    console.log(`✅ 歷史遮罩疊加層已添加: ${maskId}, 透明度: ${window.maskOpacity || 0.7}`);
}

// 🔧 新增：移除歷史遮罩疊加層（超強版）
function removeHistoryMaskOverlays(targetElement) {
    if (!targetElement) return;

    console.log('🧹 開始清除遮罩疊加層，目標元素:', targetElement.id);

    // 移除所有類型的遮罩疊加層
    const overlaySelectors = [
        '.history-mask-overlay',
        '.mask-overlay',
        '[id*="history-mask"]',
        '[id*="mask-layer"]',
        '[class*="mask"]'
    ];

    let removedCount = 0;
    overlaySelectors.forEach(selector => {
        const overlays = targetElement.querySelectorAll(selector);
        overlays.forEach(overlay => {
            console.log('🗑️ 移除遮罩疊加層:', {
                className: overlay.className,
                id: overlay.id,
                tagName: overlay.tagName
            });
            overlay.remove();
            removedCount++;
        });
    });

    // 強制清除所有子元素中可能的遮罩元素
    const allChildren = Array.from(targetElement.children);
    allChildren.forEach(child => {
        if (child.style.backgroundImage && child.style.backgroundImage.includes('masks')) {
            console.log('🗑️ 移除具有遮罩背景的子元素:', child);
            child.remove();
            removedCount++;
        }
    });

    console.log(`✅ 遮罩清除完成，共移除 ${removedCount} 個元素`);
}

// 🔧 新增：歷史模式專用的遮罩疊加函數 (PNG)
function addHistoryPngMaskOverlay(targetElement, maskPath, maskId) {
    console.log(`🎭 開始添加歷史遮罩疊加層: ${maskId}, 路徑: ${maskPath}`);

    // 移除現有的遮罩疊加層
    removeHistoryMaskOverlays(targetElement);

    // 檢查目標元素是否存在
    if (!targetElement) {
        console.error('❌ 目標元素不存在:', targetElement);
        return;
    }

    // 創建遮罩疊加層
    const overlay = document.createElement('div');
    overlay.className = 'mask-overlay';
    overlay.id = maskId;
    overlay.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url(${API_BASE_URL}/files/${maskPath});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        opacity: ${window.maskOpacity || 0.7};
        pointer-events: none;
        z-index: 2;
    `;

    // 添加到目標元素
    targetElement.appendChild(overlay);

    console.log(`✅ 歷史遮罩疊加層已添加: ${maskId}, 透明度: ${window.maskOpacity || 0.7}`);
    console.log(`📍 目標元素子節點數量:`, targetElement.children.length);
}

// 🔧 新增：移除歷史遮罩疊加層
function removeHistoryMaskOverlays(targetElement) {
    const overlays = targetElement.querySelectorAll('.mask-overlay');
    overlays.forEach(overlay => overlay.remove());
}

// 🔧 新增：測試函數 - 手動觸發歷史遮罩顯示
window.testHistoryMasks = function() {
    console.log('🧪 測試歷史遮罩顯示');
    console.log('  - window.masksVisible:', window.masksVisible);
    console.log('  - window.currentMaskType:', window.currentMaskType);

    // 強制設定遮罩為可見狀態
    window.masksVisible = true;

    // 調用載入函數
    if (typeof window.loadImagesWithMasks === 'function') {
        window.loadImagesWithMasks();
    } else {
        console.error('❌ window.loadImagesWithMasks 函數不存在');
    }
};

// 🔧 新增：圖片載入測試函數
function testImageLoad(url, description) {
    const img = new Image();
    img.onload = function() {
        console.log(`✅ ${description} 圖片載入成功:`, url);
        console.log(`  - 尺寸: ${this.naturalWidth}x${this.naturalHeight}`);
    };
    img.onerror = function() {
        console.error(`❌ ${description} 圖片載入失敗:`, url);
        // 嘗試直接訪問看看伺服器回應
        fetch(url)
            .then(response => {
                console.log(`🔍 ${description} HTTP狀態:`, response.status, response.statusText);
                return response.text();
            })
            .then(text => {
                console.log(`📄 ${description} 回應內容:`, text.substring(0, 200));
            })
            .catch(error => {
                console.error(`🚫 ${description} 網路錯誤:`, error);
            });
    };
    img.src = url;
}// ===== 歷史查看器功能已整合到主要的互動式檢視器中 =====
// 所有歷史查看功能現在使用與即時分析相同的 createInteractiveViewer 和 initializeInteractiveViewer

// ===== 簡化的歷史查看器函數（已棄用，保留供參考） =====

// 這些函數已被新的統一互動式檢視器取代
// setupHistorySliderViewer, loadHistoryImages, initializeHistorySlider 等函數已棄用

// ===== 設置歷史拉桿檢視器 =====
function setupHistorySliderViewer(data, runId) {
    const beforeImage = document.getElementById('beforeImage');
    const afterImage = document.getElementById('afterImage');
    const slider = document.getElementById('comparisonSlider');
    const opacitySlider = document.getElementById('historyMaskOpacity');
    const opacityValue = document.getElementById('historyOpacityValue');

    console.log('setupHistorySliderViewer called with runId:', runId);

    // 設置歷史圖片
    beforeImage.src = `${API_BASE_URL}/files/${runId}/upload/image1.jpg`;
    afterImage.src = `${API_BASE_URL}/files/${runId}/upload/image2.jpg`;

    console.log('Setting before image:', beforeImage.src);
    console.log('Setting after image:', afterImage.src);

    // 初始狀態（顯示原始圖片）
    let currentMaskType = 'original';

    // 拉桿控制
    if (slider) {
        slider.addEventListener('input', function() {
            const value = this.value;
            afterImage.style.clipPath = `inset(0 ${100-value}% 0 0)`;
            document.querySelector('.slider-line').style.left = `${value}%`;
        });
    }

    // 透明度控制
    if (opacitySlider) {
        opacitySlider.addEventListener('input', function() {
            const opacity = this.value;
            opacityValue.textContent = `${opacity}%`;

            // 更新遮罩透明度
            afterImage.style.opacity = opacity / 100;
        });
    }

    // 遮罩類型切換
    const maskRadios = document.querySelectorAll('input[name="historyMaskType"]');
    maskRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                currentMaskType = this.value;
                updateHistoryMaskDisplay(currentMaskType, runId);
            }
        });
    });

    // 初始化遮罩顯示
    updateHistoryMaskDisplay(currentMaskType, runId);
}

// ===== 更新歷史遮罩顯示 =====
function updateHistoryMaskDisplay(maskType, runId) {
    const afterImage = document.getElementById('afterImage');

    console.log('updateHistoryMaskDisplay called with maskType:', maskType, 'runId:', runId);

    switch(maskType) {
        case 'original':
            afterImage.src = `${API_BASE_URL}/files/${runId}/upload/image2.jpg`;
            break;
        case 'same':
            afterImage.src = `${API_BASE_URL}/files/${runId}/detection/same_objects_mask.jpg`;
            break;
        case 'disappeared':
            afterImage.src = `${API_BASE_URL}/files/${runId}/detection/disappeared_objects_mask.jpg`;
            break;
        case 'appeared':
            afterImage.src = `${API_BASE_URL}/files/${runId}/detection/appeared_objects_mask.jpg`;
            break;
    }

    console.log('Updated after image src:', afterImage.src);
}

// ===== 設置歷史物件檢視器 =====
function setupHistoryObjectViewer(data, runId) {
    // 從資料中提取物件資訊
    const objectsData = {
        disappeared: data.results?.disappeared_objects || [],
        appeared: data.results?.appeared_objects || []
    };

    let currentObjectType = 'disappeared';
    let currentObjectIndex = 0;

    // 設置物件類型切換按鈕
    const typeButtons = document.querySelectorAll('.object-type-btn');
    typeButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            // 更新按鈕狀態
            typeButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            // 切換物件類型
            currentObjectType = this.dataset.type;
            currentObjectIndex = 0;
            updateHistoryObjectDisplay(objectsData, currentObjectType, currentObjectIndex, runId);
        });
    });

    // 設置物件導航按鈕
    document.getElementById('prevObjectBtn').addEventListener('click', function() {
        const objects = objectsData[currentObjectType];
        if (objects.length > 0) {
            currentObjectIndex = (currentObjectIndex - 1 + objects.length) % objects.length;
            updateHistoryObjectDisplay(objectsData, currentObjectType, currentObjectIndex, runId);
        }
    });

    document.getElementById('nextObjectBtn').addEventListener('click', function() {
        const objects = objectsData[currentObjectType];
        if (objects.length > 0) {
            currentObjectIndex = (currentObjectIndex + 1) % objects.length;
            updateHistoryObjectDisplay(objectsData, currentObjectType, currentObjectIndex, runId);
        }
    });

    // 設置遮罩切換開關
    const maskToggle = document.getElementById('objectMaskToggle');
    const toggleSwitch = document.querySelector('.toggle-switch');
    const toggleHandle = document.querySelector('.toggle-handle');

    maskToggle.addEventListener('change', function() {
        const beforeMask = document.getElementById('objectBeforeMask');
        const afterMask = document.getElementById('objectAfterMask');

        if (this.checked) {
            toggleSwitch.style.background = '#4CAF50';
            toggleHandle.style.left = '26px';
            if (beforeMask) beforeMask.style.display = 'block';
            if (afterMask) afterMask.style.display = 'block';
        } else {
            toggleSwitch.style.background = '#ddd';
            toggleHandle.style.left = '2px';
            if (beforeMask) beforeMask.style.display = 'none';
            if (afterMask) afterMask.style.display = 'none';
        }
    });

    // 初始化物件顯示
    updateHistoryObjectDisplay(objectsData, currentObjectType, currentObjectIndex, runId);
}

// ===== 更新歷史物件顯示 =====
function updateHistoryObjectDisplay(objectsData, objectType, objectIndex, runId) {
    const objects = objectsData[objectType];
    const objectStats = document.getElementById('objectStats');
    const objectCounter = document.getElementById('objectCounter');
    const objectTitle = document.getElementById('objectTitle');

    // 更新計數器
    objectCounter.textContent = objects.length > 0 ? `${objectIndex + 1} / ${objects.length}` : '0 / 0';

    if (objects.length === 0) {
        objectTitle.textContent = `No ${objectType === 'disappeared' ? 'disappeared' : 'appeared'} objects`;
        objectStats.innerHTML = '<div style="text-align:center;color:#666;">無物件資料</div>';
        return;
    }

    const currentObject = objects[objectIndex];

    // 更新標題和統計資訊
    objectTitle.textContent = `${objectType === 'disappeared' ? 'Disappeared' : 'Appeared'} Object #${objectIndex + 1}`;

    objectStats.innerHTML = `
        <div style="display: flex; justify-content: space-between;">
            <span>Class:</span>
            <strong>${currentObject.class || 'Unknown'}</strong>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Confidence:</span>
            <strong>${(currentObject.confidence * 100).toFixed(1)}%</strong>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Area:</span>
            <strong>${currentObject.area || 'N/A'}</strong>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>位置:</span>
            <strong>[${currentObject.bbox?.join(', ') || 'N/A'}]</strong>
        </div>
    `;

    // 更新圖片
    const beforeImage = document.getElementById('objectBeforeImage');
    const afterImage = document.getElementById('objectAfterImage');
    const beforeMask = document.getElementById('objectBeforeMask');
    const afterMask = document.getElementById('objectAfterMask');

    // 設置物件圖片路徑
    beforeImage.src = `${API_BASE_URL}/file/uploads/image1.jpg`;
    afterImage.src = `${API_BASE_URL}/file/uploads/image2.jpg`;

    // 設置遮罩圖片路徑
    beforeMask.src = `${API_BASE_URL}/file/runs/${runId}/${objectType}_objects_mask.jpg`;
    afterMask.src = `${API_BASE_URL}/file/runs/${runId}/${objectType}_objects_mask.jpg`;
}

// ===== 設置歷史視覺化展示 =====
function setupHistoryVisualization(data, runId) {
    const visualizationGrid = document.getElementById('visualizationImageGrid');

    // 視覺化圖片列表
    const visualizationImages = [
        { name: 'detected_objects_1.jpg', title: '前圖檢測結果' },
        { name: 'detected_objects_2.jpg', title: '後圖檢測結果' },
        { name: 'same_objects_mask.jpg', title: 'Same Objects Mask' },
        { name: 'disappeared_objects_mask.jpg', title: 'Disappeared Objects Mask' },
        { name: 'appeared_objects_mask.jpg', title: 'Appeared Objects Mask' }
    ];

    let gridHTML = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 15px;">';

    visualizationImages.forEach(img => {
        gridHTML += `
            <div class="visualization-item" style="text-align: center; background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h6 style="margin: 0 0 10px 0; color: #333;">${img.title}</h6>
                <img src="${API_BASE_URL}/file/runs/${runId}/${img.name}"
                     style="width: 100%; height: 200px; object-fit: contain; border-radius: 6px; border: 1px solid #eee;"
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                <div style="display: none; height: 200px; background: #f8f9fa; border-radius: 6px; border: 1px solid #eee; display: flex; align-items: center; justify-content: center; color: #666;">圖片未找到</div>
            </div>
        `;
    });

    gridHTML += '</div>';
    visualizationGrid.innerHTML = gridHTML;
}

// ===== 載入歷史圖片 =====
function loadHistoryImages(data, runId) {
    // 這個函數已被新的setupHistorySliderViewer取代
    console.log('loadHistoryImages已棄用，使用setupHistorySliderViewer');
}

// ===== 初始化歷史拉桿控制 =====
function initializeHistorySlider(data, runId) {
    // 這個函數已被新的setupHistorySliderViewer取代
    console.log('initializeHistorySlider已棄用，使用setupHistorySliderViewer');
}

// ===== 初始化歷史物件檢視器 =====
function initializeHistoryObjectViewer(data, runId) {
    // 這個函數已被新的setupHistoryObjectViewer取代
    console.log('initializeHistoryObjectViewer已棄用，使用setupHistoryObjectViewer');
}

// ===== 顯示歷史物件（已棄用） =====
function showHistoryObjects(type, data, runId) {
    // 這個函數已被新的updateHistoryObjectDisplay取代
    console.log('showHistoryObjects已棄用，使用updateHistoryObjectDisplay');
}

// ===== 載入歷史全圖檢視（已棄用） =====
function loadHistoryFullView(data, runId) {
    // 這個函數已被新的setupHistoryVisualization取代
    console.log('loadHistoryFullView已棄用，使用setupHistoryVisualization');
}

// ===== 渲染歷史詳細內容（已棄用） =====
function renderHistoryDetailContent(data, runId) {
    // 這個函數已被新的renderHistoryAsLiveResults取代
    console.log('renderHistoryDetailContent已棄用，使用renderHistoryAsLiveResults');
    return '';
}

// ===== 渲染物件列表（已棄用） =====
function renderObjectsList(objects, title, type) {
    // 這個函數已被整合到新的檢視器中
    console.log('renderObjectsList已棄用');
    return '';
}

// 🎨 現代化樣式注入
function injectModernStyles() {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        /* 現代化切換開關樣式 */
        .toggle-switch {
            position: relative;
            transition: all 0.3s ease;
            background: #ddd;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }

        .toggle-handle {
            position: absolute;
            top: 2px;
            left: 2px;
            width: 22px;
            height: 22px;
            background: white;
            border-radius: 50%;
            transition: all 0.3s ease;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }

        /* 當checkbox被選中時的狀態 */
        input[type="checkbox"]:checked + label .toggle-switch {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
        }

        input[type="checkbox"]:checked + label .toggle-switch .toggle-handle {
            transform: translateX(24px) !important;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4) !important;
        }

        /* 按鈕懸停效果增強 */
        .modern-button {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .modern-button:before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }

        .modern-button:hover:before {
            left: 100%;
        }

        /* 卡片陰影動畫 */
        .stat-card {
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
        }

        /* 圖片容器懸停效果 */
        .image-container {
            transition: all 0.3s ease;
        }

        .image-container:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15) !important;
        }

        /* 進度條樣式 */
        .progress-bar {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 4px;
            border-radius: 2px;
            transition: width 0.3s ease;
        }

        /* 漸變文字效果 */
        .gradient-text {
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
    `;
    document.head.appendChild(styleElement);
}

// ===== 保留您原有的全域變數 =====
// 💾 Save Parameterslet currentMode = 'advanced';
let selectedPhotos = [];
let selectedVideos = [];
let currentImageIndex = 0;
let previewImages = [];
let isProcessing = false;
let currentSessionId = null;
let segmentationResult = null; // 儲存分割結果
let detectionResults = null; // 儲存檢測結果
let selectedImagePair = [null, null]; // 儲存使用者選擇的兩張圖片索引

// 🔧 新增：拉桿檢視器相關變數
let sliderPosition = 50;
let masksVisible = false;
let currentMaskType = 'same';
let maskOpacity = 0.7;
let separatedImages = null;

// 🔧 新增：物件檢視器相關變數
let currentObjectType = 'disappeared';
let currentObjectIndex = 0;
let objectMaskVisible = false; // 記住物件檢視器的遮罩顯示狀態
let objectsData = {
    disappeared: [],
    appeared: []
};

// API 基礎 URL
const API_BASE_URL = 'http://127.0.0.1:5000/api';

// 🔧 新增：檔案服務 URL 統一管理
const FILE_SERVICE_URL = `${API_BASE_URL}/files`;

// ===== 輔助函數 =====
// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// ===== Loading Overlay 函數 =====
function showLoadingOverlay(message = 'Processing...') {
    // 移除現有的 overlay（如果存在）
    hideLoadingOverlay();

    const overlay = document.createElement('div');
    overlay.id = 'loadingOverlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        color: white;
        font-size: 18px;
    `;

    overlay.innerHTML = `
        <div class="loading" style="width: 40px; height: 40px; border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 20px;"></div>
        <div>${message}</div>
    `;

    document.body.appendChild(overlay);
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.remove();
    }
}

// ===== 保留您原有的初始化函式 =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('Photo Change Detection System loaded - Supports slider view');
    console.log('初始 currentSessionId:', currentSessionId);
    initializeSystem();
    setupEventListeners();
    setupDragAndDrop();
});

function showAlert(message, type = 'info') {
    console.log(`${getAlertIcon(type)} ${message}`);

    // 創建簡單的通知顯示
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px;
        border-radius: 5px;
        z-index: 10000;
        max-width: 400px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;
        font-size: 14px;
    `;

    // 設定不同類型的樣式
    switch(type) {
        case 'success':
            alertDiv.style.background = '#d4edda';
            alertDiv.style.color = '#155724';
            alertDiv.style.border = '1px solid #c3e6cb';
            break;
        case 'error':
            alertDiv.style.background = '#f8d7da';
            alertDiv.style.color = '#721c24';
            alertDiv.style.border = '1px solid #f5c6cb';
            break;
        case 'warning':
            alertDiv.style.background = '#fff3cd';
            alertDiv.style.color = '#856404';
            alertDiv.style.border = '1px solid #ffeaa7';
            break;
        default:
            alertDiv.style.background = '#d1ecf1';
            alertDiv.style.color = '#0c5460';
            alertDiv.style.border = '1px solid #bee5eb';
    }

    alertDiv.textContent = message;
    document.body.appendChild(alertDiv);

    // 3秒後自動移除
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.parentNode.removeChild(alertDiv);
        }
    }, 3000);
}

// 2. getAlertIcon 輔助函數
function getAlertIcon(type) {
    switch(type) {
        case 'success': return '✅';
        case 'error': return '❌';
        case 'warning': return '⚠️';
        default: return 'ℹ️';
    }
}

// 3. setupDragAndDrop 函數
function setupDragAndDrop() {
    console.log('🔧 Initialize drag and drop upload function...');

    // 查找上傳區域元素
    const uploadArea = document.getElementById('your-actual-upload-id') ||
                      document.querySelector('.your-actual-upload-class') ||
                      document.querySelector('#your-specific-element') ||
                      document.querySelector('body'); // 備用：使用整個頁面

    if (!uploadArea) {
        console.warn('⚠️ 找不到上傳區域元素，跳過拖拽設定');
        return;
    }

    // 防止瀏覽器預設行為
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // 拖拽進入和離開的視覺效果
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    // 檔案拖放處理
    uploadArea.addEventListener('drop', handleDrop, false);

    console.log('✅ Drag and drop upload function initialization complete');

    // 內部函數定義
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        uploadArea.classList.add('drag-over');
        if (uploadArea.style) {
            uploadArea.style.backgroundColor = '#f0f8ff';
            uploadArea.style.borderColor = '#007bff';
        }
    }

    function unhighlight(e) {
        uploadArea.classList.remove('drag-over');
        if (uploadArea.style) {
            uploadArea.style.backgroundColor = '';
            uploadArea.style.borderColor = '';
        }
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            console.log(`📁 拖拽上傳 ${files.length} 個檔案`);
            handleFiles(files);
        }
    }

    // 處理檔案的函數（如果不存在則建立簡單版本）
    function handleFiles(files) {
        if (typeof handleFileUpload === 'function') {
            // 如果有現成的檔案處理函數
            for (let file of files) {
                handleFileUpload(file);
            }
        } else {
            // 簡單的檔案處理
            console.log('📄 Files detected:', Array.from(files).map(f => f.name));
            showAlert(`Detected ${files.length} files, please implement file processing logic`, 'info');
        }
    }
}

// 4. 初始化系統函數的改進版本
async function initializeSystem() {
    try {
        console.log('🚀 System initialization started...');

        // 🎨 注入現代化樣式
        injectModernStyles();

        // 🔧 載入並初始化參數
        loadParametersFromStorage();

        // 檢查後端連線
        await checkBackendConnection();

        // 設定拖拽上傳
        setupDragAndDrop();

        // 初始化為進階模式
        setMode('advanced');

        // 設定其他初始化...
        // setupEventListeners(); // 如果有其他事件監聽器

        showAlert('System initialization successful!', 'success');
        console.log('✅ 系統初始化完成');

    } catch (error) {
        console.error('系統初始化失敗:', error);
        showAlert(`系統初始化失敗: ${error.message}`, 'error');
    }
}

// 5. 檢查後端連線函數
async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            throw new Error(`後端服務異常 (HTTP ${response.status})`);
        }

        const data = await response.json();
        console.log('✅ Backend connection normal:', data);
        return true;

    } catch (error) {
        console.warn('⚠️ 後端連線檢查失敗:', error);
        throw error;
    }
}

// 事件監聽
function setupEventListeners() {
    document.getElementById('photoInput').addEventListener('change', handlePhotoSelection);
    document.getElementById('videoInput').addEventListener('change', handleVideoSelection);
    document.getElementById('photoFolder').addEventListener('change', (e) => handleFolderSelection(e, 'photo'));
    document.getElementById('videoFolder').addEventListener('change', (e) => handleFolderSelection(e, 'video'));

    // 只保留Advanced按鈕
    document.getElementById('advancedBtn').addEventListener('click', startAdvancedPipeline);

    document.getElementById('imageIndex').addEventListener('change', function() {
        goToImage(this.value);
    });
}

// 模式切換 (只保留Advanced模式)
function setMode(mode) {
    console.log('🔄 Set mode:', mode);
    currentMode = 'advanced'; // 強制使用Advanced模式

    // 移除所有active class
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));

    // 獲取按鈕元素
    const advancedBtn = document.getElementById('advancedBtn');

    console.log('🔍 Found Advanced button:', { advancedBtn });

    // 只顯示Advanced模式
    const advancedModeBtn = document.querySelector('.mode-btn.advanced');
    if (advancedModeBtn) {
        advancedModeBtn.classList.add('active');
    }
    if (advancedBtn) {
        advancedBtn.style.display = 'inline-block';
    }

    console.log('✅ 模式設置完成: advanced (only mode available)');
}

// 照片選擇
function handlePhotoSelection(event) {
    const files = Array.from(event.target.files);
    selectedPhotos = files.filter(file => {
        console.log(`檔案: ${file.name}, 大小: ${file.size} bytes`);
        return file.type.startsWith('image/') && file.size > 0;
    });
    if (selectedPhotos.some(file => file.size === 0)) {
        showAlert('Warning: Empty file detected, please reselect', 'warning');
    }
    // 重置選擇的圖片對
    selectedImagePair = [null, null];
    updatePhotoStatus();
    loadPreviewImages();
}

// 影片選擇
function handleVideoSelection(event) {
    const files = Array.from(event.target.files);
    selectedVideos = files.filter(file => file.type.startsWith('video/'));
    updateVideoStatus();

    if (selectedVideos.length > 0) {
        displayVideoList();
        document.getElementById('videoSelectionSection').style.display = 'block';
        document.getElementById('videoSection').style.display = 'block';
    } else {
        document.getElementById('videoSelectionSection').style.display = 'none';
        document.getElementById('videoSection').style.display = 'none';
    }
}

// 資料夾選擇
function handleFolderSelection(event, type) {
    const files = Array.from(event.target.files);
    if (type === 'photo') {
        selectedPhotos = files.filter(file => file.type.startsWith('image/'));
        selectedImagePair = [null, null]; // 重置選擇
        updatePhotoStatus();
        loadPreviewImages();
    } else if (type === 'video') {
        selectedVideos = files.filter(file => file.type.startsWith('video/'));
        updateVideoStatus();

        if (selectedVideos.length > 0) {
            displayVideoList();
            document.getElementById('videoSelectionSection').style.display = 'block';
            document.getElementById('videoSection').style.display = 'block';
        } else {
            document.getElementById('videoSelectionSection').style.display = 'none';
            document.getElementById('videoSection').style.display = 'none';
        }
    }
}

// 狀態顯示
function updatePhotoStatus() {
    const status = document.getElementById('photoStatus');
    if (selectedPhotos.length > 0) {
        status.textContent = `${selectedPhotos.length} photos selected`;
        status.style.color = '#4CAF50';
        status.style.fontWeight = 'bold';
    } else {
        status.textContent = '尚未選取照片';
        status.style.color = '#666';
        status.style.fontWeight = 'normal';
    }
}

function updateVideoStatus() {
    const status = document.getElementById('videoStatus');
    if (selectedVideos.length > 0) {
        status.textContent = `已選取 ${selectedVideos.length} 個影片`;
        status.style.color = '#4CAF50';
        status.style.fontWeight = 'bold';
    } else {
        status.textContent = '尚未選取影片';
        status.style.color = '#666';
        status.style.fontWeight = 'normal';
    }
}

// 影片處理相關函數
let selectedVideoForProcessing = null; // 用於跟蹤選擇要處理的影片

// 顯示影片列表
function displayVideoList() {
    const videoList = document.getElementById('videoList');
    if (!videoList) return;

    videoList.innerHTML = selectedVideos.map((video, index) => {
        const videoUrl = URL.createObjectURL(video);
        return `
            <div class="video-item" data-index="${index}" onclick="selectVideoForProcessing(${index})">
                <video class="video-thumbnail" src="${videoUrl}" muted preload="metadata">
                    Your browser does not support the video tag.
                </video>
                <div class="video-info">
                    <div class="video-name" style="font-weight: 600; color: #333; margin: 8px 0 4px 0; font-size: 14px; line-height: 1.3;">${video.name}</div>
                    <div class="video-size" style="color: #666; font-size: 12px;">${formatFileSize(video.size)}</div>
                </div>
            </div>
        `;
    }).join('');

    // 載入影片縮圖（取第一幀作為縮圖）
    setTimeout(() => {
        document.querySelectorAll('.video-thumbnail').forEach(video => {
            video.currentTime = 1; // 設置到第1秒以獲得縮圖
        });
    }, 100);
}

// 選擇要處理的影片
function selectVideoForProcessing(index) {
    const video = selectedVideos[index];
    if (!video) return;

    // 清除之前的選擇
    document.querySelectorAll('.video-item').forEach(item => {
        item.classList.remove('selected');
    });

    // 選擇當前影片
    const videoItem = document.querySelector(`.video-item[data-index="${index}"]`);
    if (videoItem) {
        videoItem.classList.add('selected');
    }

    // 更新選擇的影片
    selectedVideoForProcessing = { index, file: video };

    // 顯示選擇狀態
    updateVideoSelectionStatus();
}

// 更新影片選擇狀態顯示
function updateVideoSelectionStatus() {
    const statusSection = document.getElementById('videoSelectionStatus');
    const statusContent = document.getElementById('selectedVideoName');

    if (!statusSection || !statusContent) return;

    if (selectedVideoForProcessing) {
        const video = selectedVideoForProcessing.file;
        statusContent.textContent = `${video.name} (${formatFileSize(video.size)})`;
        statusSection.style.display = 'block';
    } else {
        statusContent.textContent = 'No video selected yet';
        statusSection.style.display = 'none';
    }
}

// 清除影片處理選擇
function clearVideoProcessingSelection() {
    selectedVideoForProcessing = null;

    // 清除視覺選擇
    document.querySelectorAll('.video-item').forEach(item => {
        item.classList.remove('selected');
    });

    // 更新狀態顯示
    updateVideoSelectionStatus();
}

// 預覽控制 - 支援靈活選擇
function loadPreviewImages() {
    if (selectedPhotos.length === 0) return;

    console.log('🔄 Loading preview images, selectedPhotos count:', selectedPhotos.length);

    // Convert File objects to URLs for display
    previewImages = selectedPhotos.map((file, index) => {
        if (file instanceof File) {
            console.log(`📁 Converting File ${index + 1} to URL:`, file.name);
            return URL.createObjectURL(file);
        } else {
            console.log(`🔗 File ${index + 1} is already a URL:`, file);
            return file; // Already a URL string
        }
    });

    console.log('✅ Preview images converted, count:', previewImages.length);

    currentImageIndex = 0;
    updatePreview();
    const navigationControls = document.getElementById('navigationControls');
    if (previewImages.length > 2) {
        navigationControls.style.display = 'flex';
        document.getElementById('imageIndex').max = previewImages.length - 1;
        // 如果還沒有選擇圖片對，預設選擇前兩張
        if (selectedImagePair[0] === null && selectedImagePair[1] === null) {
            selectedImagePair = [0, 1];
            console.log('🎯 自動選擇前兩張圖片:', selectedImagePair);
        }
        // 顯示圖片選擇網格
        showImageSelectionGrid();
    } else {
        navigationControls.style.display = 'none';
        // 自動選擇前兩張圖片
        if (previewImages.length >= 2) {
            selectedImagePair = [0, 1];
            console.log('🎯 自動選擇前兩張圖片 (≤2張):', selectedImagePair);
        }
        // 即使只有2張圖片也顯示選擇網格
        showImageSelectionGrid();
    }
}

// Update preview displays based on selected image pair
function updatePreviewDisplays() {
    const preview1 = document.getElementById('preview1');
    const preview2 = document.getElementById('preview2');

    if (!preview1 || !preview2) return;

    // Clear existing content
    preview1.innerHTML = '';
    preview2.innerHTML = '';
    preview1.className = '';
    preview2.className = '';

    // Display selected images
    if (selectedImagePair[0] !== null && previewImages[selectedImagePair[0]]) {
        const img1 = createSelectablePreviewElement(previewImages[selectedImagePair[0]], selectedImagePair[0]);
        preview1.appendChild(img1);
        preview1.className = 'preview-content selected';
    } else {
        preview1.className = 'no-preview';
        preview1.innerHTML = 'Image 1<br>No image selected yet';
    }

    if (selectedImagePair[1] !== null && previewImages[selectedImagePair[1]]) {
        const img2 = createSelectablePreviewElement(previewImages[selectedImagePair[1]], selectedImagePair[1]);
        preview2.appendChild(img2);
        preview2.className = 'preview-content selected';
    } else {
        preview2.className = 'no-preview';
        preview2.innerHTML = 'Image 2<br>No image selected yet';
    }
}

function updatePreview() {
    const preview1 = document.getElementById('preview1');
    const preview2 = document.getElementById('preview2');
    preview1.innerHTML = '';
    preview2.innerHTML = '';
    preview1.className = '';
    preview2.className = '';

    // 如果有選擇特定的圖片對，優先顯示
    if (selectedImagePair[0] !== null && selectedImagePair[1] !== null) {
        if (previewImages[selectedImagePair[0]]) {
            const img1 = createSelectablePreviewElement(previewImages[selectedImagePair[0]], selectedImagePair[0]);
            preview1.appendChild(img1);
            preview1.className = 'preview-content selected';
        }

        if (previewImages[selectedImagePair[1]]) {
            const img2 = createSelectablePreviewElement(previewImages[selectedImagePair[1]], selectedImagePair[1]);
            preview2.appendChild(img2);
            preview2.className = 'preview-content selected';
        }
    } else {
        // 預設顯示前兩張圖片
        if (previewImages.length > currentImageIndex) {
            const img1 = createSelectablePreviewElement(previewImages[currentImageIndex], currentImageIndex);
            preview1.appendChild(img1);
            preview1.className = 'preview-content';
        } else {
            preview1.className = 'no-preview';
            preview1.textContent = '尚未選取影像';
        }

        if (previewImages.length > currentImageIndex + 1) {
            const img2 = createSelectablePreviewElement(previewImages[currentImageIndex + 1], currentImageIndex + 1);
            preview2.appendChild(img2);
            preview2.className = 'preview-content';
        } else {
            preview2.className = 'no-preview';
            preview2.textContent = '尚未選取影像';
        }
    }

    document.getElementById('imageIndex').value = currentImageIndex + 1;
}

// 🎯 新增：更新圖片預覽（支援URL和文件）
function updateImagePreview() {
    const preview1 = document.getElementById('preview1');
    const preview2 = document.getElementById('preview2');
    preview1.innerHTML = '';
    preview2.innerHTML = '';
    preview1.className = '';
    preview2.className = '';

    // 處理選擇的圖片對
    if (selectedImagePair[0] !== null) {
        const img1Data = selectedImagePair[0];
        let imgElement1;

        if (img1Data.url) {
            // 來自URL（如影片影格）
            imgElement1 = document.createElement('img');
            imgElement1.src = img1Data.url;
            imgElement1.alt = img1Data.name || 'Image 1';
            imgElement1.style.cssText = 'width: 100%; height: 300px; object-fit: contain; border-radius: 8px;';

            const container1 = document.createElement('div');
            container1.style.textAlign = 'center';
            container1.appendChild(imgElement1);

            if (img1Data.isFromVideo) {
                const timeLabel = document.createElement('div');
                timeLabel.textContent = `影格: ${img1Data.name} (${img1Data.timestamp.toFixed(1)}s)`;
                timeLabel.style.cssText = 'margin-top: 5px; font-size: 12px; color: #666;';
                container1.appendChild(timeLabel);
            }

            preview1.appendChild(container1);
        } else if (img1Data.file || previewImages[img1Data]) {
            // 來自文件
            const file = img1Data.file || previewImages[img1Data];
            imgElement1 = createSelectablePreviewElement(file, img1Data);
            preview1.appendChild(imgElement1);
        }

        preview1.className = 'preview-content selected';
    } else {
        preview1.className = 'no-preview';
        preview1.textContent = '尚未選取影像';
    }

    if (selectedImagePair[1] !== null) {
        const img2Data = selectedImagePair[1];
        let imgElement2;

        if (img2Data.url) {
            // 來自URL（如影片影格）
            imgElement2 = document.createElement('img');
            imgElement2.src = img2Data.url;
            imgElement2.alt = img2Data.name || 'Image 2';
            imgElement2.style.cssText = 'width: 100%; height: 300px; object-fit: contain; border-radius: 8px;';

            const container2 = document.createElement('div');
            container2.style.textAlign = 'center';
            container2.appendChild(imgElement2);

            if (img2Data.isFromVideo) {
                const timeLabel = document.createElement('div');
                timeLabel.textContent = `影格: ${img2Data.name} (${img2Data.timestamp.toFixed(1)}s)`;
                timeLabel.style.cssText = 'margin-top: 5px; font-size: 12px; color: #666;';
                container2.appendChild(timeLabel);
            }

            preview2.appendChild(container2);
        } else if (img2Data.file || previewImages[img2Data]) {
            // 來自文件
            const file = img2Data.file || previewImages[img2Data];
            imgElement2 = createSelectablePreviewElement(file, img2Data);
            preview2.appendChild(imgElement2);
        }

        preview2.className = 'preview-content selected';
    } else {
        preview2.className = 'no-preview';
        preview2.textContent = '尚未選取影像';
    }
}

// 新增函式：創建可選擇的預覽元素
function createSelectablePreviewElement(file, index) {
    const container = document.createElement('div');
    container.className = 'selectable-preview';

    const img = document.createElement('img');
    img.className = 'preview-media';

    // Check if file is a File object or already a URL string
    if (file instanceof File) {
        img.src = URL.createObjectURL(file);
        img.alt = file.name;
        img.onload = () => URL.revokeObjectURL(img.src);
    } else {
        // file is already a URL string
        img.src = file;
        img.alt = `Image ${index + 1}`;
    }

    const label = document.createElement('div');
    label.className = 'image-label';
    label.textContent = `Photo ${index + 1}`;
    label.style.cssText = `
        position: absolute; top: 10px; left: 10px;
        background: rgba(76, 175, 80, 0.8); color: white;
        padding: 4px 8px; border-radius: 4px; font-weight: bold;
    `;

    container.style.position = 'relative';
    container.appendChild(img);
    container.appendChild(label);

    return container;
}

function createPreviewElement(file) {
    const element = document.createElement('img');
    element.className = 'preview-media';

    // Check if file is a File object or already a URL string
    if (file instanceof File) {
        element.src = URL.createObjectURL(file);
        element.alt = file.name;
        element.onload = () => URL.revokeObjectURL(element.src);
    } else {
        // file is already a URL string
        element.src = file;
        element.alt = 'Preview Image';
    }

    return element;
}

// 導航
function previousImage() {
    if (currentImageIndex > 0) {
        currentImageIndex--;
        updatePreview();
    }
}

function nextImage() {
    if (currentImageIndex < previewImages.length - 2) {
        currentImageIndex++;
        updatePreview();
    }
}

function goToImage(index) {
    const newIndex = parseInt(index) - 1;
    if (newIndex >= 0 && newIndex < previewImages.length - 1) {
        currentImageIndex = newIndex;
        updatePreview();
    }
}

function goToFrame() {
    const index = parseInt(document.getElementById('imageIndex').value);
    goToImage(index);
}

// 🎥 新增：從URL獲取圖片作為Blob
async function fetchImageAsBlob(imageUrl) {
    try {
        const response = await fetch(imageUrl);
        if (!response.ok) {
            throw new Error(`無法載入圖片: ${response.status}`);
        }
        return await response.blob();
    } catch (error) {
        console.error('載入圖片失敗:', error);
        throw new Error(`載入圖片失敗: ${error.message}`);
    }
}

// 修改檔案上傳函式使用選擇的圖片
async function uploadFiles() {
    // 🎥 修正：支援影片影格上傳
    if (selectedImagePair[0] === null || selectedImagePair[1] === null) {
        if (previewImages.length >= 2) {
            // 如果沒有特定選擇，使用預設的前兩張
            selectedImagePair = [0, 1];
        } else {
            throw new Error('請選擇兩張圖片進行比較');
        }
    }

    const formData = new FormData();

    // 🔧 如果已有會話ID，傳遞給後端以重用現有run
    if (currentSessionId) {
        formData.append('session_id', currentSessionId);
        console.log(`♻️ 重用現有會話: ${currentSessionId}`);
    }

    // 🎥 檢查是否為影片影格（包含URL）
    if (typeof selectedImagePair[0] === 'object' && selectedImagePair[0].url) {
        // 從影片影格：需要先下載圖片再上傳
        console.log('📤 Processing video frame upload...');

        const image1Blob = await fetchImageAsBlob(selectedImagePair[0].url);
        const image2Blob = await fetchImageAsBlob(selectedImagePair[1].url);

        formData.append('ref_image', image1Blob, selectedImagePair[0].name);
        formData.append('input_image', image2Blob, selectedImagePair[1].name);

        console.log(`📤 正在上傳影格: ${selectedImagePair[0].name} vs ${selectedImagePair[1].name}`);
    } else {
        // 從檔案：需要獲取原始 File 對象
        if (!selectedPhotos[selectedImagePair[0]] || !selectedPhotos[selectedImagePair[1]]) {
            throw new Error('需要兩個圖片檔案');
        }

        formData.append('ref_image', selectedPhotos[selectedImagePair[0]]);
        formData.append('input_image', selectedPhotos[selectedImagePair[1]]);

        console.log(`📤 正在上傳: 圖片${selectedImagePair[0] + 1} vs 圖片${selectedImagePair[1] + 1}`);
        console.log(`📁 檔案1: ${selectedPhotos[selectedImagePair[0]].name}`);
        console.log(`📁 檔案2: ${selectedPhotos[selectedImagePair[1]].name}`);
    }

    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'File upload failed');
    }
    const result = await response.json();
    currentSessionId = result.session_id;

    // 儲存運行編號給物件檢視使用 - 從 run_id 解析
    if (result.run_id) {
        // run_id 格式為 "run_030"，取出後面的數字
        const match = result.run_id.match(/run_(\d+)/);
        if (match) {
            window.currentRunNumber = parseInt(match[1], 10);
            console.log('💾 儲存運行編號:', window.currentRunNumber, '來自 run_id:', result.run_id);
        }
    }

    console.log('✅ File upload successful, Session ID:', currentSessionId);
    return result;
}



// 進階 AI 流程 - 🔧 修改為支援分離圖片和影片影格
async function startAdvancedPipeline() {
    // 🔧 修復：執行前先更新參數從HTML表單
    console.log('📋 更新分析參數從HTML表單...');
    saveParameters();
    console.log('✅ 當前分析參數:', analysisParameters);

    // 🎥 修正：檢查是否有選擇的圖片（來自檔案或影片影格）
    const hasValidImages = (previewImages.length >= 2) ||
                          (selectedImagePair[0] !== null && selectedImagePair[1] !== null);

    console.log('🔍 檢查圖片狀態:');
    console.log('  - previewImages.length:', previewImages.length);
    console.log('  - selectedImagePair:', selectedImagePair);
    console.log('  - selectedPhotos.length:', selectedPhotos.length);
    console.log('  - hasValidImages:', hasValidImages);

    if (!hasValidImages) {
        showAlert('請選擇至少兩張圖片進行檢測', 'warning');
        return;
    }
    if (isProcessing) {
        showAlert('Processing, please wait...', 'warning');
        return;
    }
    isProcessing = true;
    document.getElementById('progressSection').style.display = 'block';

    // 重置結果
    segmentationResult = null;
    detectionResults = null;
    separatedImages = null;

    let stepResults = {
        uploadResult: null,
        alignResult: null,
        skyRemovalResult: null,  // 🆕 新增天空遮罩步驟
        segmentResult: null,
        matchResult: null,
        changeResult: null
    };

    try {
        // 步驟 1：上傳圖像檔案
        console.log('🚀 Starting step 1: Upload image files');
        stepResults.uploadResult = await executeStep(1, '上傳圖像檔案', uploadFiles);

        if (!currentSessionId && stepResults.uploadResult?.session_id) {
            currentSessionId = stepResults.uploadResult.session_id;
        }

        // 步驟 2：執行圖像對齊
        console.log('🚀 Starting step 2: Execute image alignment');
        stepResults.alignResult = await executeStep(2, '執行圖像對齊', () => alignImages(currentSessionId));

        // 🆕 步驟 3：天空遮罩去除 (use parameter setting)
        console.log('🚀 Starting step 3: Sky mask removal');
        const enableSkyRemoval = analysisParameters.enableSkyRemoval;
        stepResults.skyRemovalResult = await executeStep(3, '天空遮罩去除', () => removeSkyMasks(currentSessionId, enableSkyRemoval));

        // 步驟 4：執行 SAM2 語意分割（使用參數設定）
        console.log('🚀 Starting step 4: Execute SAM2 semantic segmentation');
        stepResults.segmentResult = await executeStep(4, '執行 SAM2 語意分割', () => segmentImages(currentSessionId, analysisParameters));

        segmentationResult = stepResults.segmentResult;

        // 步驟 5：執行遮罩匹配（使用參數設定）
        console.log('🚀 Starting step 5: Execute mask matching');
        stepResults.matchResult = await executeStep(5, '執行遮罩匹配', () => matchMasks(currentSessionId, stepResults.segmentResult, analysisParameters));

        // 步驟 6：執行變化檢測（使用參數設定）
        console.log('🚀 Starting step 6: Execute change detection');
        stepResults.changeResult = await executeStep(6, '執行變化檢測', () => detectChanges(currentSessionId, analysisParameters));

        // 儲存檢測結果
        detectionResults = stepResults.changeResult;

        // 處理分離圖片結果
        await processSeparatedImagesResults();

        // 顯示結果
        showAdvancedResults();
        showAlert('進階 AI 檢測完成！已整合天空遮罩去除功能', 'success');

    } catch (error) {
        console.error('💥 進階檢測失敗:', error);
        showAlert('進階檢測過程發生錯誤: ' + error.message, 'error');
        resetProcessingState();
    } finally {
        isProcessing = false;
    }
}

// 🆕 新增：天空遮罩去除函數

async function removeSkyMasks(sessionId, enableSkyRemoval = true) {
    console.log('🌤️ 步驟 3: 天空遮罩去除開始', { enableSkyRemoval });

    try {
        const params = {
            session_id: sessionId || currentSessionId,
            device: 'auto',  // 可以根據需要調整
            enable_sky_removal: enableSkyRemoval  // 🔧 新增：天空遮罩去除開關
        };

        console.log('🌤️ 發送天空遮罩去除請求:', params);

        const response = await fetch(`${API_BASE_URL}/remove_sky`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `天空遮罩去除失敗 (${response.status})`);
        }

        const result = await response.json();
        console.log('✅ 天空遮罩去除完成:', result);

        return result;

    } catch (error) {
        console.error('💥 天空遮罩去除失敗:', error);
        throw error;
    }
}

async function executeStep(stepNumber, stepName, stepFunction) {
    console.log(`🔄 Starting step ${stepNumber}: ${stepName}`);
    updateStepStatus(stepNumber, 'active', 'Processing...');
    try {
        const result = await stepFunction();
        console.log(`✅ Step ${stepNumber} completed:`, result);
        updateStepStatus(stepNumber, 'completed', 'Complete');
        return result;
    } catch (error) {
        console.error(`❌ 步驟 ${stepNumber} 失敗:`, error);
        updateStepStatus(stepNumber, 'error', '失敗: ' + error.message);
        throw error;
    }
}

function updateStepStatus(stepNumber, status, message) {
    const step = document.getElementById(`step${stepNumber}`);
    const icon = step.querySelector('.step-icon');
    const statusText = step.querySelector('.step-status');
    step.classList.remove('active', 'completed', 'error');
    icon.classList.remove('pending', 'active', 'completed', 'error');
    step.classList.add(status);
    icon.classList.add(status);
    statusText.textContent = message;
    if (status === 'completed') icon.textContent = '✓';
    else if (status === 'error') icon.textContent = '✗';
    else if (status === 'active') icon.innerHTML = '<div class="loading"></div>';
    else icon.textContent = stepNumber;
}

// API 調用
async function alignImages() {
    if (!currentSessionId) throw new Error('工作階段ID不存在，請重新上傳檔案');
    const params = {
        session_id: currentSessionId,
        motion_type: 'EUCLIDEAN',  // 預設使用歐式變換
        pyramid_levels: 4          // 預設金字塔層數
    };
    console.log('🔧 發送對齊參數:', params);
    const response = await fetch(`${API_BASE_URL}/align`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    });
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || '圖像對齊失敗');
    }
    return await response.json();
}

// SAM2 分割函式 - 使用固定最佳參數
async function segmentImages(sessionId, parameters = null) {
    console.log('🔄 步驟 4: SAM2 分割開始');
    console.log('🔑 使用工作階段ID:', sessionId || currentSessionId);
    console.log('📋 傳入的參數物件:', parameters);

    // 🔧 修復：確保 sessionId 參數正確傳遞
    const activeSessionId = sessionId || currentSessionId;
    if (!activeSessionId) {
        throw new Error('工作階段ID不存在，請重新上傳檔案');
    }

    try {
        // 使用傳入的參數或默認參數
        const segmentParams = parameters ? {
            points_per_side: parameters.pointsPerSide,
            points_per_batch: parameters.pointsPerBatch,
            pred_iou_thresh: parameters.predIouThresh,
            stability_score_thresh: parameters.stabilityScoreThresh,
            stability_score_offset: parameters.stabilityScoreOffset,
            min_mask_region_area: parameters.minMaskRegionArea
        } : {};

        console.log('🔧 構建的分割參數:', segmentParams);
        console.log('🔍 詳細參數檢查:');
        console.log('  - 原始 pointsPerSide:', parameters?.pointsPerSide);
        console.log('  - 轉換後 points_per_side:', segmentParams.points_per_side);

        const params = {
            session_id: activeSessionId,
            device: 'auto',
            ...segmentParams
        };

        console.log('🤖 發送分割請求（使用自定義參數）:', params);
        console.log('🌐 即將發送到後端的完整參數:', JSON.stringify(params, null, 2));

        const response = await fetch(`${API_BASE_URL}/segment`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error('❌ SAM2 分割 API 錯誤:', errorData);
            throw new Error(errorData.message || `SAM2 分割失敗 (HTTP ${response.status})`);
        }

        const result = await response.json();

        // 🔧 關鍵修復：詳細記錄結果結構以供遮罩匹配使用
        console.log('✅ SAM2 分割完成');
        console.log('📊 分割結果狀態:', result.status);

        if (result.status === 'success' && result.data) {
            console.log('📁 分割結果詳細資訊:');
            console.log('  - 輸出目錄:', result.data.output_directory);
            console.log('  - 處理圖像數:', result.data.processed_images);
            console.log('  - 總遮罩數:', result.data.num_masks);

            // 🔧 重要：檢查並記錄 results 陣列結構
            if (result.data.results && Array.isArray(result.data.results)) {
                console.log(`📋 找到 ${result.data.results.length} 個圖像處理結果:`);

                result.data.results.forEach((item, index) => {
                    console.log(`  結果 ${index + 1}:`);
                    console.log(`    - 圖像路徑: ${item.result?.original_image_path}`);
                    console.log(`    - 遮罩檔案: ${item.result?.masks_pickle_path}`);
                    console.log(`    - 遮罩數量: ${item.result?.num_masks}`);
                    console.log(`    - 輸出目錄: ${item.result?.output_directory}`);
                });

                // 🔧 驗證遮罩檔案路徑完整性
                const validResults = result.data.results.filter(item =>
                    item.result?.masks_pickle_path && item.result?.original_image_path
                );

                if (validResults.length < 2) {
                    console.warn('⚠️ 警告：找到的有效遮罩檔案少於2個，可能影響後續匹配');
                    console.log('有效結果數量:', validResults.length);
                } else {
                    console.log('✅ 找到足夠的遮罩檔案供匹配使用');
                }
            } else {
                console.warn('⚠️ 警告：分割結果中沒有 results 陣列');
                console.log('可用的數據鍵:', Object.keys(result.data));
            }

            // 🔧 新增：保存到全域變數供後續步驟使用
            segmentationResult = result;
            console.log('💾 已儲存分割結果供遮罩匹配使用');

            // 🔧 新增：如果只有單一結果，嘗試特殊處理
            if (!result.data.results && result.data.masks_pickle_path) {
                console.log('🔧 檢測到單一結果模式');
                console.log('  - 單一遮罩檔案:', result.data.masks_pickle_path);

                // 創建兼容的結果結構
                result.data.results = [{
                    result: {
                        original_image_path: result.data.original_image_path,
                        masks_pickle_path: result.data.masks_pickle_path,
                        num_masks: result.data.num_masks,
                        output_directory: result.data.output_directory
                    }
                }];
                console.log('🔧 已轉換為標準結果格式');
            }
        } else {
            console.error('❌ 分割結果狀態異常:', result);
            throw new Error('分割結果狀態異常或無數據');
        }

        console.log('🔍 最終分割結果結構:', JSON.stringify(result, null, 2));
        console.log('✅ 已使用系統最佳分割參數，無需手動調整');

        return result;

    } catch (error) {
        console.error('💥 SAM2 分割失敗:', error);

        // 🔧 提供更詳細的錯誤信息
        if (error.message.includes('fetch')) {
            throw new Error('網絡連接失敗，請檢查後端服務是否正常運行');
        } else if (error.message.includes('JSON')) {
            throw new Error('後端響應格式錯誤，請檢查服務器狀態');
        } else {
            throw error;
        }
    }
}

async function matchMasks(sessionId, segmentResult, parameters = null) {
    console.log('🔄 步驟 5: 遮罩匹配開始');
    console.log('📊 接收到的分割結果:', segmentResult?.status);

    try {
        const activeSegmentResult = segmentResult || segmentationResult;

        if (!activeSegmentResult || activeSegmentResult.status !== 'success') {
            throw new Error('分割結果無效或不完整');
        }

        let masks1Path = null;
        let masks2Path = null;
        let image1Path = null;
        let image2Path = null;

        // 🔧 修復：適配新的結果結構
        if (activeSegmentResult.data?.results && Array.isArray(activeSegmentResult.data.results)) {
            const results = activeSegmentResult.data.results;
            console.log(`📊 分析 ${results.length} 個分割結果`);

            if (results.length >= 2) {
                // 🆕 關鍵修正：構建 all_masks 目錄路徑
                const result1 = results[0].result;
                const result2 = results[1].result;

                if (result1?.output_directory && result2?.output_directory) {
                    // 🆕 確保路徑格式統一
                    const outputDir1 = result1.output_directory.replace(/\//g, '\\');
                    const outputDir2 = result2.output_directory.replace(/\//g, '\\');

                    masks1Path = `${outputDir1}\\single_pass_masks`;
                    masks2Path = `${outputDir2}\\single_pass_masks`;

                    console.log('📁 修正後的檔案路徑:');
                    console.log(`  - 遮罩目錄1: ${masks1Path}`);
                    console.log(`  - 遮罩目錄2: ${masks2Path}`);
                } else {
                    throw new Error('分割結果中缺少 output_directory');
                }
            } else {
                throw new Error(`分割結果數量不足：需要2個，實際得到${results.length}個`);
            }
        } else {
            throw new Error('分割結果格式錯誤：缺少 results 陣列');
        }

        // 驗證路徑完整性
        if (!masks1Path || !masks2Path) {
            throw new Error('無法從分割結果中提取完整的遮罩檔案路徑');
        }

        // 其他程式碼保持不變...
        // 使用傳入的參數或默認參數
        const matchParams = parameters ? {
            iou_threshold: parameters.iouThreshold,
            distance_threshold: parameters.distanceThreshold,
            similarity_threshold: parameters.similarityThreshold
        } : {
            iou_threshold: 0.2,
            distance_threshold: 50,
            similarity_threshold: 0.25
        };

        const requestData = {
            session_id: sessionId || currentSessionId,
            masks_1_path: masks1Path,
            masks_2_path: masks2Path,
            image1_path: image1Path,
            image2_path: image2Path,
            ...matchParams
        };

        console.log('📤 發送遮罩匹配請求:', requestData);

        const response = await fetch(`${API_BASE_URL}/match_masks`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error('❌ 遮罩匹配 API 錯誤:', errorData);
            throw new Error(errorData.message || `遮罩匹配失敗 (HTTP ${response.status})`);
        }

        const result = await response.json();
        console.log('✅ 遮罩匹配完成:', result);

        return result;

    } catch (error) {
        console.error('💥 遮罩匹配失敗:', error);
        throw error;
    }
}


// 🔧 修正：變化檢測函式 - 支援新API格式並啟用拉桿檢視器
async function detectChanges(sessionId = null, parameters = null) {
    if (!currentSessionId && !sessionId) throw new Error('工作階段ID不存在，請重新上傳檔案');

    const params = {
        session_id: sessionId || currentSessionId,
        normalized_diff_threshold: 0.10      // 預設標準化差異閾值
    };

    // 嘗試從分割結果中獲取圖像路徑
    if (segmentationResult && segmentationResult.data && segmentationResult.data.results) {
        const results = segmentationResult.data.results;
        console.log('🔍 從分割結果獲取圖像路徑, 結果數量:', results.length);

        if (results.length >= 2) {
            if (results[0].image_path && results[1].image_path) {
                params.image_path_old = results[0].image_path;
                params.image_path_new = results[1].image_path;
                console.log('✅ 成功獲取圖像路徑:');
                console.log('   - 舊圖像:', params.image_path_old);
                console.log('   - 新圖像:', params.image_path_new);
            } else {
                console.log('⚠️ 分割結果中缺少圖像路徑，使用 session_id 模式');
            }
        } else {
            console.log('⚠️ 分割結果數量不足，使用 session_id 模式');
        }
    } else {
        console.log('⚠️ 沒有分割結果，使用 session_id 模式');
    }

    console.log('🔍 發送變化檢測參數:', params);

    // 🔧 新增：智慧重試機制
    const maxRetries = 3;
    let lastError;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            console.log(`🚀 變化檢測嘗試 ${attempt}/${maxRetries}`);

            const response = await fetch(`${API_BASE_URL}/detect_change`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || `變化檢測失敗 (${response.status})`);
            }

            const result = await response.json();
            console.log('✅ 變化檢測成功:', result);

            // 🔧 關鍵修正：兼容新舊API格式並啟用拉桿檢視器
            if (result.success) {
                console.log('✅ 變化檢測完成');

                // 🔧 兼容新的API格式
                const imageData = result.data?.generated_images || result.data?.separated_images || result.generated_images;

                if (imageData) {
                    // 🆕 重新啟用拉桿檢視器
                    separatedImages = imageData; // 更新全域變數

                    // 初始化拉桿檢視器
                    initializeSliderViewer(imageData);

                    // 顯示拉桿容器
                    const comparisonContainer = document.getElementById('comparison-container');
                    if (comparisonContainer) {
                        comparisonContainer.style.display = 'block';
                    }

                    console.log('✅ 拉桿檢視器已重新啟用');
                } else {
                    console.error('❌ 無法找到圖片資料');
                }

                // 🔧 保持原有的結果儲存邏輯
                detectionResults = result;
                console.log('💾 已儲存檢測結果供網頁展示');

                return result;
            }

        } catch (error) {
            lastError = error;
            console.warn(`⚠️ 嘗試 ${attempt} 失敗:`, error.message);

            if (attempt < maxRetries) {
                console.log(`🔄 等待 ${attempt * 2} 秒後重試...`);
                await new Promise(resolve => setTimeout(resolve, attempt * 2000));
            }
        }
    }

    console.error('❌ 所有重試都失敗了');
    throw lastError;
}

// 🔧 新增：拉桿檢視器初始化函數
function initializeSliderViewer(imageData) {
    console.log('初始化拉桿檢視器:', imageData);

    if (!imageData) {
        console.error('❌ 圖片資料為空，無法初始化拉桿檢視器');
        return;
    }

    // 更新全域變數
    separatedImages = imageData;

    // 預設顯示原始圖片比較
    updateSliderImages('original');

    // 顯示拉桿控制區域
    const sliderSection = document.getElementById('sliderSection');
    if (sliderSection) {
        sliderSection.style.display = 'block';
    }

    // 重新初始化拉桿控制項
    setupSliderControls();
}

// 🔧 新增：設定拉桿控制項
function setupSliderControls() {
    // 確保拉桿容器存在
    const comparisonContainer = document.getElementById('comparison-container');
    if (comparisonContainer) {
        comparisonContainer.style.display = 'block';
    }

    // 重置拉桿位置
    sliderPosition = 50;
    updateSliderPosition();

    // 確保遮罩類型選擇器事件綁定
    const maskTypeSelector = document.getElementById('maskType');
    if (maskTypeSelector) {
        maskTypeSelector.removeEventListener('change', handleMaskTypeChange);
        maskTypeSelector.addEventListener('change', handleMaskTypeChange);
    }
}

// 🔧 新增：更新拉桿圖片
function updateSliderImages(type) {
    if (!separatedImages) {
        console.error('❌ separatedImages 為空');
        return;
    }

    const leftImage = document.getElementById('leftImage');
    const rightImage = document.getElementById('rightImage');

    if (!leftImage || !rightImage) {
        console.error('❌ 找不到拉桿圖片元素');
        return;
    }

    let leftSrc, rightSrc;

    switch(type) {
        case 'original':
            leftSrc = separatedImages.image1_original;
            rightSrc = separatedImages.image2_original;
            break;
        case 'same':
            leftSrc = separatedImages.image1_same_masks;
            rightSrc = separatedImages.image2_same_masks;
            break;
        case 'disappeared':
            leftSrc = separatedImages.image1_disappeared_masks;
            rightSrc = separatedImages.image1_original;
            break;
        case 'appeared':
            leftSrc = separatedImages.image2_appeared_masks;
            rightSrc = separatedImages.image2_original;
            break;
        default:
            console.error('❌ 未知的圖片類型:', type);
            return;
    }

    // 🔧 確保使用正確的檔案服務 URL
    if (leftSrc) {
        leftImage.src = `${FILE_SERVICE_URL}/${leftSrc.replace(/^.*[\\\/]/, '')}`;
        console.log('左圖設定:', leftImage.src);
    }

    if (rightSrc) {
        rightImage.src = `${FILE_SERVICE_URL}/${rightSrc.replace(/^.*[\\\/]/, '')}`;
        console.log('右圖設定:', rightImage.src);
    }

    currentMaskType = type;
}

// 🔧 新增：更新拉桿位置
function updateSliderPosition() {
    // 拉桿位置更新邏輯
    console.log('更新拉桿位置:', sliderPosition);
}

// 🔧 新增：處理遮罩類型變更
function handleMaskTypeChange(event) {
    const selectedType = event.target.value;
    updateSliderImages(selectedType);
}

// 🔧 新增：處理6張分離圖片結果
async function processSeparatedImagesResults() {
    console.log('🖼️ 處理6張分離圖片結果...');

    if (detectionResults && detectionResults.success) {
        const changeData = detectionResults.data;

        // 🔧 修正：兼容新舊API格式
        const imageData = changeData?.generated_images || changeData?.separated_images;

        // 儲存分離圖片資訊
        if (imageData) {
            separatedImages = imageData;
            console.log('📸 分離圖片資訊:', separatedImages);
        }

        // 儲存物件檢視資料
        if (imageData) {
            console.log('🔍 檢查物件檢視資料:', imageData);

            if (imageData.disappeared_objects) {
                objectsData.disappeared = imageData.disappeared_objects;
                console.log('📤 Disappeared objects count:', objectsData.disappeared.length);
            }
            if (imageData.appeared_objects) {
                objectsData.appeared = imageData.appeared_objects;
                console.log('📥 New objects count:', objectsData.appeared.length);
            }

            // 如果沒有物件資料，嘗試從其他地方獲取
            if (!imageData.disappeared_objects && !imageData.appeared_objects) {
                console.log('⚠️ 沒有在imageData中找到物件資料，檢查changeData結構:', changeData);

                // 嘗試從檢測結果中提取物件資料
                if (changeData.analysis_results) {
                    const results = changeData.analysis_results;
                    objectsData.disappeared = results.disappeared_objects || [];
                    objectsData.appeared = results.appeared_objects || [];
                    console.log('✅ Extract object data from analysis_results');
                    console.log('📤 Disappeared objects count:', objectsData.disappeared.length);
                    console.log('📥 New objects count:', objectsData.appeared.length);
                }
            }
        }

        console.log('🎯 分離圖片處理完成');

        // 更新物件檢視顯示
        console.log('🔄 Update object view, disappeared objects:', objectsData.disappeared.length, 'new objects:', objectsData.appeared.length);
        await updateObjectDisplay();
    } else {
        console.warn('⚠️ 未找到變化檢測結果');
    }
}

// 🔧 新增：重置處理狀態
function resetProcessingState() {
    segmentationResult = null;
    detectionResults = null;
    separatedImages = null;
    objectsData = { disappeared: [], appeared: [] };
}

// 影格提取
async function extractFrames() {
    if (selectedVideos.length === 0) {
        showAlert('請先選擇影片檔案', 'warning');
        return;
    }

    const interval = parseFloat(document.getElementById('frameInterval').value);
    const maxFrames = 50; // 預設最大影格數

    // 顯示載入狀態
    const extractBtn = document.querySelector('button[onclick="extractFrames()"]');
    const originalText = extractBtn.textContent;
    extractBtn.textContent = 'Processing...';
    extractBtn.disabled = true;

    try {
        showAlert('開始處理影片，請稍候...', 'info');

        // 只處理第一個選中的影片
        const videoFile = selectedVideos[0];

        // 準備FormData
        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('interval', interval.toString());
        formData.append('max_frames', maxFrames.toString());

        // 🔧 新增：如果有現有會話，傳遞session_id
        if (currentSessionId) {
            formData.append('session_id', currentSessionId);
            console.log('🔄 使用現有會話ID:', currentSessionId);
        }

        // 發送請求到後端
        const response = await fetch(`${API_BASE_URL}/extract_frames`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.status === 'success') {
            const data = result.data;

            // 🔧 新增：更新當前會話ID
            if (data.session_id && !currentSessionId) {
                currentSessionId = data.session_id;
                console.log('✅ 設置會話ID:', currentSessionId);
            }

            showAlert(
                `影片處理完成！提取了 ${data.extracted_frames} 個影格\n` +
                `影片長度: ${data.video_info.duration.toFixed(2)}秒\n` +
                `FPS: ${data.video_info.fps.toFixed(1)}`,
                'success'
            );

            // 🎯 修改：顯示影格提取結果到新位置
            displayVideoFrames(data);

            // 不再顯示在檢測結果區域，而是在影格區域
            // document.getElementById('resultsSection').style.display = 'block';

        } else {
            showAlert(`影片處理失敗: ${result.message}`, 'error');
        }

    } catch (error) {
        console.error('影片處理錯誤:', error);
        showAlert('影片處理時發生錯誤', 'error');
    } finally {
        // 恢復按鈕狀態
        extractBtn.textContent = originalText;
        extractBtn.disabled = false;
    }
}

// 新的函數：從選定的影片提取影格
async function extractFramesFromSelectedVideo() {
    if (!selectedVideoForProcessing) {
        showAlert('請先選擇要處理的影片', 'warning');
        return;
    }

    const videoFile = selectedVideoForProcessing.file;

    const interval = parseFloat(document.getElementById('frameInterval').value);
    const maxFrames = 50; // 預設最大影格數

    // 顯示載入狀態
    const extractBtn = document.querySelector('button[onclick="extractFramesFromSelectedVideo()"]');
    const originalText = extractBtn.textContent;
    extractBtn.textContent = 'Processing...';
    extractBtn.disabled = true;

    try {
        showAlert(`開始處理影片: ${videoFile.name}，請稍候...`, 'info');

        // 準備FormData
        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('interval', interval.toString());
        formData.append('max_frames', maxFrames.toString());

        // 🔧 新增：如果有現有會話，傳遞session_id
        if (currentSessionId) {
            formData.append('session_id', currentSessionId);
            console.log('🔄 使用現有會話ID:', currentSessionId);
        }

        // 發送請求到後端
        const response = await fetch(`${API_BASE_URL}/extract_frames`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.status === 'success') {
            const data = result.data;

            // 🔧 新增：更新當前會話ID
            if (data.session_id && !currentSessionId) {
                currentSessionId = data.session_id;
                console.log('✅ 設置會話ID:', currentSessionId);
            }

            showAlert(
                `影片處理完成！\n` +
                `影片: ${videoFile.name}\n` +
                `提取了 ${data.extracted_frames} 個影格\n` +
                `影片長度: ${data.video_info.duration.toFixed(2)}秒\n` +
                `FPS: ${data.video_info.fps.toFixed(1)}`,
                'success'
            );

            // 顯示影格提取結果
            displayVideoFrames(data);

        } else {
            showAlert(`影片處理失敗: ${result.message}`, 'error');
        }

    } catch (error) {
        console.error('影片處理錯誤:', error);
        showAlert('影片處理時發生錯誤', 'error');
    } finally {
        // 恢復按鈕狀態
        extractBtn.textContent = originalText;
        extractBtn.disabled = false;
    }
}

// 🎥 新增：顯示影片處理結果 - 專注於影格選擇
function displayVideoResults(data) {
    const resultsContent = document.getElementById('resultsContent');

    resultsContent.innerHTML = `
        <div class="result-item">
            <h4>🎥 影格提取結果</h4>
            <div class="result-stats">
                <div class="stat-item">
                    <div class="stat-value">${data.extracted_frames}</div>
                    <div class="stat-label">提取影格</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${data.video_info.duration.toFixed(1)}s</div>
                    <div class="stat-label">影片長度</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${data.video_info.fps.toFixed(1)}</div>
                    <div class="stat-label">FPS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">Run ${data.run_number}</div>
                    <div class="stat-label">運行編號</div>
                </div>
            </div>

            <div style="margin-top: 20px;">
                <h5>📽️ 點擊選擇2個影格進行變化檢測</h5>
                <p style="color: #666; margin-bottom: 15px;">
                    ${data.message || '請選擇任意2個影格來進行變化檢測分析'}
                </p>

                <div class="video-frames-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 15px; max-height: 400px; overflow-y: auto;">
                    ${data.frames.map((frame, index) => `
                        <div class="frame-item clickable-frame"
                             data-frame-url="${API_BASE_URL}/files/results/runs/run_${String(data.run_number).padStart(3, '0')}/video_processing/frames/${frame.filename}"
                             data-frame-name="${frame.filename}"
                             data-frame-timestamp="${frame.timestamp}"
                             onclick="selectFrameForAnalysis(this)"
                             style="text-align: center; border: 2px solid #ddd; padding: 8px; border-radius: 8px; cursor: pointer; transition: all 0.3s;">
                            <img src="${API_BASE_URL}/files/results/runs/run_${String(data.run_number).padStart(3, '0')}/video_processing/frames/${frame.filename}"
                                 style="width: 100%; height: 100px; object-fit: cover; border-radius: 5px;"
                                 alt="Frame ${index + 1}">
                            <div style="font-size: 11px; margin-top: 8px; color: #333; font-weight: bold;">
                                影格 ${index + 1}
                            </div>
                            <div style="font-size: 10px; color: #666;">
                                ${frame.timestamp.toFixed(1)}s
                            </div>
                        </div>
                    `).join('')}
                </div>

                <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                        <span style="font-weight: bold;">已選擇影格:</span>
                        <div id="selectedFramesDisplay" style="color: #666;">
                            尚未選擇影格
                        </div>
                    </div>
                    <button id="analyzeSelectedFrames"
                            class="action-btn btn-primary"
                            onclick="startFrameAnalysis()"
                            disabled
                            style="opacity: 0.5;">
                        開始分析選中的影格
                    </button>
                </div>
            </div>
        </div>
    `;

    // 初始化影格選擇狀態
    window.selectedFramesForAnalysis = [];
}

// 🎥 新增：在專用區域顯示影格提取結果
function displayVideoFrames(data) {
    const videoFramesSection = document.getElementById('videoFramesSection');
    const videoFramesContent = document.getElementById('videoFramesContent');

    if (!videoFramesSection || !videoFramesContent) {
        console.error('找不到影格顯示區域');
        return;
    }

    videoFramesContent.innerHTML = `
        <div class="video-info" style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; text-align: center;">
                <div>
                    <div style="font-size: 18px; font-weight: bold; color: #2196F3;">${data.extracted_frames}</div>
                    <div style="font-size: 12px; color: #666;">提取影格</div>
                </div>
                <div>
                    <div style="font-size: 18px; font-weight: bold; color: #4CAF50;">${data.video_info.duration.toFixed(1)}s</div>
                    <div style="font-size: 12px; color: #666;">影片長度</div>
                </div>
                <div>
                    <div style="font-size: 18px; font-weight: bold; color: #FF9800;">${data.video_info.fps.toFixed(1)}</div>
                    <div style="font-size: 12px; color: #666;">FPS</div>
                </div>
                <div>
                    <div style="font-size: 18px; font-weight: bold; color: #9C27B0;">Run ${data.run_number}</div>
                    <div style="font-size: 12px; color: #666;">運行編號</div>
                </div>
            </div>
        </div>

        <div style="margin-bottom: 15px;">
            <h4 style="margin: 0 0 10px 0;">📽️ 點擊選擇2個影格進行變化檢測</h4>
            <p style="color: #666; margin: 0; font-size: 14px;">
                ${data.message || '請選擇任意2個影格來進行變化檢測分析'}
            </p>
        </div>

        <div class="video-frames-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; max-height: 350px; overflow-y: auto; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px;">
            ${data.frames.map((frame, index) => `
                <div class="frame-item clickable-frame"
                     data-frame-url="${API_BASE_URL}/files/results/runs/run_${String(data.run_number).padStart(3, '0')}/video_processing/frames/${frame.filename}"
                     data-frame-name="${frame.filename}"
                     data-frame-timestamp="${frame.timestamp}"
                     onclick="selectFrameForAnalysis(this)"
                     style="text-align: center; border: 2px solid #ddd; padding: 6px; border-radius: 6px; cursor: pointer; transition: all 0.3s; background: white;">
                    <img src="${API_BASE_URL}/files/results/runs/run_${String(data.run_number).padStart(3, '0')}/video_processing/frames/${frame.filename}"
                         style="width: 100%; height: 80px; object-fit: cover; border-radius: 4px;"
                         alt="Frame ${index + 1}">
                    <div style="font-size: 11px; margin-top: 6px; color: #333; font-weight: bold;">
                        影格 ${index + 1}
                    </div>
                    <div style="font-size: 10px; color: #666;">
                        ${frame.timestamp.toFixed(1)}s
                    </div>
                </div>
            `).join('')}
        </div>

        <div style="padding: 15px; background: #f8f9fa; border-radius: 8px;">
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 12px;">
                <span style="font-weight: bold;">已選擇影格:</span>
                <div id="selectedFramesDisplay" style="color: #666; flex: 1;">
                    尚未選擇影格
                </div>
            </div>
            <button id="analyzeSelectedFrames"
                    class="action-btn btn-primary"
                    onclick="startFrameAnalysis()"
                    disabled
                    style="opacity: 0.5; width: 100%;">
                開始分析選中的影格
            </button>
        </div>
    `;

    // 顯示影格區域
    videoFramesSection.style.display = 'block';

    // 初始化影格選擇狀態
    window.selectedFramesForAnalysis = [];

    // 滾動到影格區域
    videoFramesSection.scrollIntoView({ behavior: 'smooth' });
}

// 🎯 新增：影格選擇功能
function selectFrameForAnalysis(frameElement) {
    const frameUrl = frameElement.dataset.frameUrl;
    const frameName = frameElement.dataset.frameName;
    const frameTimestamp = parseFloat(frameElement.dataset.frameTimestamp);

    // 檢查是否已選擇
    const existingIndex = window.selectedFramesForAnalysis.findIndex(f => f.url === frameUrl);

    if (existingIndex >= 0) {
        // 取消選擇
        window.selectedFramesForAnalysis.splice(existingIndex, 1);
        frameElement.style.border = '2px solid #ddd';
        frameElement.style.background = 'white';
    } else {
        // 選擇影格
        if (window.selectedFramesForAnalysis.length >= 2) {
            showAlert('最多只能選擇2個影格進行比較', 'warning');
            return;
        }

        window.selectedFramesForAnalysis.push({
            url: frameUrl,
            name: frameName,
            timestamp: frameTimestamp,
            element: frameElement
        });

        // 視覺化回饋
        frameElement.style.border = '2px solid #4CAF50';
        frameElement.style.background = '#f0f8f0';
    }

    updateSelectedFramesDisplay();
}

// 🎯 更新選擇的影格顯示
function updateSelectedFramesDisplay() {
    const display = document.getElementById('selectedFramesDisplay');
    const analyzeBtn = document.getElementById('analyzeSelectedFrames');

    if (window.selectedFramesForAnalysis.length === 0) {
        display.textContent = '尚未選擇影格';
        analyzeBtn.disabled = true;
        analyzeBtn.style.opacity = '0.5';
    } else if (window.selectedFramesForAnalysis.length === 1) {
        const frame = window.selectedFramesForAnalysis[0];
        display.textContent = `已選擇1個影格: ${frame.name} (${frame.timestamp.toFixed(1)}s)`;
        analyzeBtn.disabled = true;
        analyzeBtn.style.opacity = '0.5';
    } else if (window.selectedFramesForAnalysis.length === 2) {
        const frame1 = window.selectedFramesForAnalysis[0];
        const frame2 = window.selectedFramesForAnalysis[1];
        display.innerHTML = `
            已選擇2個影格:<br>
            1. ${frame1.name} (${frame1.timestamp.toFixed(1)}s)<br>
            2. ${frame2.name} (${frame2.timestamp.toFixed(1)}s)
        `;
        analyzeBtn.disabled = false;
        analyzeBtn.style.opacity = '1';
    }
}

// 🎯 開始影格分析
function startFrameAnalysis() {
    if (window.selectedFramesForAnalysis.length !== 2) {
        showAlert('請選擇2個影格進行分析', 'warning');
        return;
    }

    const frame1 = window.selectedFramesForAnalysis[0];
    const frame2 = window.selectedFramesForAnalysis[1];

    // 🔗 將選擇的影格設定為分析用的影像對
    selectedImagePair = [
        {
            file: null,
            url: frame1.url,
            name: frame1.name,
            isFromVideo: true,
            timestamp: frame1.timestamp
        },
        {
            file: null,
            url: frame2.url,
            name: frame2.name,
            isFromVideo: true,
            timestamp: frame2.timestamp
        }
    ];

    // 更新圖片預覽
    updateImagePreview();

    // 自動切換到進階分析模式
    setMode('advanced');

    // 顯示成功訊息
    const timeDiff = Math.abs(frame2.timestamp - frame1.timestamp);
    showAlert(
        `影格分析準備完成！\n` +
        `選擇的影格時間間隔: ${timeDiff.toFixed(1)}秒\n` +
        `現在可以進行變化檢測分析`,
        'success'
    );

    // 滾動到圖片預覽區域
    document.getElementById('previewSection').scrollIntoView({ behavior: 'smooth' });
}

// 🎥 載入影片影格進行進階分析
async function loadVideoFramesForAnalysis(runNumber) {
    try {
        showAlert('載入影片影格中...', 'info');

        const response = await fetch(`${API_BASE_URL}/video_frames/${runNumber}`);
        const result = await response.json();

        if (result.status === 'success') {
            const frames = result.data.frames;

            if (frames.length < 2) {
                showAlert('需要至少2個影格才能進行分析', 'warning');
                return;
            }

            // 自動選擇第一個和最後一個影格進行分析
            const firstFrame = frames[0];
            const lastFrame = frames[frames.length - 1];

            // 模擬選擇影像對
            selectedImagePair = [
                {
                    file: null,
                    url: firstFrame.url,
                    name: firstFrame.filename
                },
                {
                    file: null,
                    url: lastFrame.url,
                    name: lastFrame.filename
                }
            ];

            // 更新預覽
            updateImagePreview();

            showAlert(
                `已選擇影格進行分析:\n` +
                `起始: ${firstFrame.filename} (${firstFrame.timestamp.toFixed(1)}s)\n` +
                `結束: ${lastFrame.filename} (${lastFrame.timestamp.toFixed(1)}s)`,
                'success'
            );

            // 自動切換到照片分析模式
            document.getElementById('analysisMode').value = 'advanced';

        } else {
            showAlert(`載入影格失敗: ${result.message}`, 'error');
        }

    } catch (error) {
        console.error('載入影格錯誤:', error);
        showAlert('載入影格時發生錯誤', 'error');
    }
}



// 🔧 修改：完全重寫 showAdvancedResults 函式 - 支援拉桿檢視器
function showAdvancedResults() {
    const resultsContent = document.getElementById('resultsContent');

    if (!detectionResults || !detectionResults.data) {
        resultsContent.innerHTML = '<div class="result-item"><p>⚠️ 無法獲取檢測結果</p></div>';
        document.getElementById('resultsSection').style.display = 'block';
        return;
    }

    const data = detectionResults.data;
    const summary = data.analysis_summary || {};

    // 從segmentationResult獲取遮罩數量
    let totalMasks = 0;
    if (segmentationResult && segmentationResult.data && segmentationResult.data.results) {
        segmentationResult.data.results.forEach(result => {
            if (result.result && result.result.num_masks) {
                totalMasks += result.result.num_masks;
            }
        });
    }

    // 🔧 新增：創建互動式檢視器
    let viewerHTML = '';
    if (separatedImages) {
        viewerHTML = createInteractiveViewer(data);
    }

    resultsContent.innerHTML = `
        <div class="result-item">
            ${viewerHTML}
        </div>
    `;

    // 🔧 初始化拉桿檢視器功能
    if (separatedImages) {
        initializeInteractiveViewer();
    }

    // 顯示結果圖片
    displayResultImages(data);

    document.getElementById('resultsSection').style.display = 'block';
}

// 🔧 新增：創建互動式檢視器HTML
function createInteractiveViewer(data) {
    return `
        <div style="margin-top: 25px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-left: 4px solid #667eea; padding: 20px; border-radius: 10px;">
            <h4 style="color: #667eea; margin-bottom: 20px;">🖼️ Interactive Image Viewer</h4>

            <!-- 分頁選擇 -->
            <div style="display: flex; background: #e9ecef; border-radius: 8px; padding: 4px; margin-bottom: 20px;">
                <button class="viewer-tab active" onclick="switchViewerTab('slider')"
                        style="flex: 1; padding: 10px; border: none; background: #667eea; color: white; border-radius: 6px; cursor: pointer; transition: all 0.3s;">
                    🎬 Slider View
                </button>
                <button class="viewer-tab" onclick="switchViewerTab('objects')"
                        style="flex: 1; padding: 10px; border: none; background: transparent; color: #333; border-radius: 6px; cursor: pointer; transition: all 0.3s;">
                    🔍 Object View
                </button>
            </div>

            <!-- Slider View Tab -->
            <div id="sliderViewerTab" class="viewer-content">
                <div style="position: relative; width: 100%; height: 400px; border-radius: 8px; overflow: hidden; background: #f0f0f0; border: 2px solid #ddd; margin-bottom: 15px;">
                    <div id="imageLayer1" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-size: contain; background-position: center; background-repeat: no-repeat; z-index: 1;"></div>
                    <div id="imageLayer2" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-size: contain; background-position: center; background-repeat: no-repeat; z-index: 2; clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%);">
                    </div>
                    <div id="sliderHandle" style="position: absolute; top: 0; left: 50%; width: 4px; height: 100%; background: #667eea; cursor: ew-resize; z-index: 10; transform: translateX(-50%); box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);">
                        <div style="position: absolute; top: 50%; left: 50%; width: 20px; height: 40px; background: #667eea; border-radius: 8px; cursor: ew-resize; transform: translate(-50%, -50%); display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">⇄</div>
                    </div>
                </div>

                <!-- 控制面板 -->
                <div style="padding: 15px; background: #f8f9fa; border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <label>🎭 Mask Display:</label>
                        <div class="toggle-switch" id="maskToggle" onclick="toggleMasks()" style="width: 50px; height: 25px; background: #ddd; border-radius: 15px; cursor: pointer; position: relative; transition: background 0.3s;">
                            <div class="toggle-handle" style="position: absolute; top: 2px; left: 2px; width: 21px; height: 21px; background: white; border-radius: 50%; transition: transform 0.3s;"></div>
                        </div>
                    </div>

                    <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                        <button class="mask-type-btn active" onclick="selectMaskType('same')" style="padding: 6px 12px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Same Objects</button>
                        <button class="mask-type-btn" onclick="selectMaskType('different')" style="padding: 6px 12px; background: #e9ecef; color: #333; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Different Objects</button>
                    </div>

                    <div>
                        <label for="opacitySlider">Transparency: <span id="opacityValue">70%</span></label>
                        <input type="range" id="opacitySlider" min="0" max="100" value="70" style="width: 100%;" onchange="updateOpacity(this.value)">
                    </div>
                </div>
            </div>

            <!-- Object View Tab -->
            <div id="objectsViewerTab" class="viewer-content" style="display: none;">
                <div style="display: flex; background: #e9ecef; border-radius: 8px; padding: 4px; margin-bottom: 15px;">
                    <button class="object-tab active" onclick="switchObjectType('disappeared')" style="flex: 1; padding: 8px; border: none; background: #667eea; color: white; border-radius: 6px; cursor: pointer;">📤 Disappeared Objects</button>
                    <button class="object-tab" onclick="switchObjectType('appeared')" style="flex: 1; padding: 8px; border: none; background: transparent; color: #333; border-radius: 6px; cursor: pointer;">📥 Appeared Objects</button>
                </div>

                <!-- 🎨 現代化遮罩控制區域 -->
                <div style="background: linear-gradient(135deg, #f8f9fa, #ffffff); border-radius: 15px; padding: 18px; margin-bottom: 20px; border: 1px solid #e9ecef; box-shadow: 0 3px 12px rgba(0,0,0,0.05);">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div style="width: 6px; height: 6px; background: #667eea; border-radius: 50%; box-shadow: 0 0 8px rgba(102, 126, 234, 0.4);"></div>
                            <h4 style="margin: 0; font-size: 16px; font-weight: 700; color: #2c3e50; letter-spacing: -0.3px;">
                                Mask Display Control
                            </h4>
                        </div>

                        <!-- 現代化切換開關 -->
                        <label for="objectMaskToggle" style="display: flex; align-items: center; gap: 10px; cursor: pointer;">
                            <input type="checkbox" id="objectMaskToggle" onchange="toggleMaskDisplay()"
                                   style="display: none;">
                            <div class="toggle-switch" style="width: 50px; height: 26px; border-radius: 13px;">
                                <div class="toggle-handle"></div>
                            </div>
                            <span style="font-size: 14px; font-weight: 600; color: #495057;">🎭 Show Mask</span>
                        </label>
                    </div>

                    <div style="font-size: 13px; color: #6c757d; padding: 10px; background: #f1f3f4; border-radius: 8px; border-left: 3px solid #667eea;">
                        <span style="font-weight: 600;">Tip:</span>
                        Disappeared objects show <span style="color: #dc3545; font-weight: 600;">red mask</span>,
                        appeared objects show <span style="color: #28a745; font-weight: 600;">green mask</span>
                    </div>
                </div>

                <div style="width: 100%; border-radius: 8px; overflow: hidden; background: #f0f0f0; border: 2px solid #ddd; margin-bottom: 15px;" id="objectDisplay">
                    <!-- 物件圖片顯示區域 - 使用與正常分析相同的水平佈局 -->
                    <div style="display: flex; width: 100%; height: 300px; gap: 15px; background: #f8f9fa; border-radius: 12px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">

                        <!-- 圖片區域 - 佔左側較大空間 -->
                        <div style="flex: 3; display: flex; gap: 15px; height: 100%;">
                            <!-- Before Change Image -->
                            <div class="image-container" style="width: 50%; height: 100%; position: relative; overflow: hidden; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                                <img id="objectBeforeImage" style="width: 100%; height: 100%; object-fit: contain; object-position: center; background: white;" alt="Before Change - Local Area">
                                <canvas id="objectBeforeMask" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: none; opacity: 0.7; z-index: 3; pointer-events: none;" alt="Before Change Mask Overlay"></canvas>
                                <div style="position: absolute; top: 12px; left: 12px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);">
                                    📷 Before Change
                                </div>
                            </div>

                            <!-- 分隔線 -->
                            <div style="width: 2px; background: linear-gradient(to bottom, #667eea, #764ba2); border-radius: 1px; opacity: 0.6;"></div>

                            <!-- After Change Image -->
                            <div class="image-container" style="width: 50%; height: 100%; position: relative; overflow: hidden; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                                <img id="objectAfterImage" style="width: 100%; height: 100%; object-fit: contain; object-position: center; background: white;" alt="After Change - Local Area">
                                <canvas id="objectAfterMask" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: none; opacity: 0.7; z-index: 3; pointer-events: none;" alt="After Change Mask Overlay"></canvas>
                                <div style="position: absolute; top: 12px; left: 12px; background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);">
                                    📷 After Change
                                </div>
                            </div>
                        </div>

                        <!-- 資訊區域 - 佔右側空間 -->
                        <div class="stats-card" style="flex: 1; display: flex; flex-direction: column; height: 100%;">
                            <div id="objectStatusText" style="text-align: center; color: #666; padding: 20px;">
                                等待檢測結果...
                            </div>
                        </div>
                    </div>
                </div>

                <div style="display: flex; justify-content: space-between; align-items: center; padding: 15px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px; box-shadow: 0 3px 12px rgba(0,0,0,0.05);">

                    <!-- Previous Button -->
                    <button onclick="previousObject()" id="prevObjectBtn"
                            class="modern-button"
                            style="display: flex; align-items: center; gap: 8px; padding: 12px 20px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; border-radius: 25px; cursor: pointer; font-weight: 600; font-size: 14px; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);"
                            onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(102, 126, 234, 0.4)'"
                            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(102, 126, 234, 0.3)'">
                        <span style="font-size: 16px;">⬅️</span>
                        <span>Previous</span>
                    </button>

                    <!-- 物件計數器 -->
                    <div style="background: white; padding: 10px 20px; border-radius: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border: 2px solid #e9ecef;">
                        <span id="objectInfo" class="gradient-text" style="font-weight: 700; font-size: 16px; letter-spacing: 0.5px;">0 / 0</span>
                    </div>

                    <!-- Next Button -->
                    <button onclick="nextObject()" id="nextObjectBtn"
                            class="modern-button"
                            style="display: flex; align-items: center; gap: 8px; padding: 12px 20px; background: linear-gradient(135deg, #28a745, #20c997); color: white; border: none; border-radius: 25px; cursor: pointer; font-weight: 600; font-size: 14px; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);"
                            onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(40, 167, 69, 0.4)'"
                            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(40, 167, 69, 0.3)'">
                        <span>Next</span>
                        <span style="font-size: 16px;">➡️</span>
                    </button>
                </div>
            </div>
        </div>
    `;
}

// 🔧 新增：初始化互動式檢視器
async function initializeInteractiveViewer() {
    console.log('🎮 初始化互動式檢視器...');

    // 🔧 確保遮罩類型已初始化（特別是歷史模式）
    if (!window.currentMaskType) {
        window.currentMaskType = 'different'; // 預設顯示差異物件
        console.log('🎭 設置預設遮罩類型為: different');
    }

    // 初始化拉桿檢視器
    initializeSliderViewer();

    // 載入原始圖片
    loadOriginalImages();

    // 初始化物件檢視器
    await updateObjectDisplay();

    // 🔧 確保遮罩類型按鈕狀態正確
    setTimeout(() => {
        const sameBtns = document.querySelectorAll('button[onclick="selectMaskType(\'same\')"]');
        const differentBtns = document.querySelectorAll('button[onclick="selectMaskType(\'different\')"]');

        sameBtns.forEach(btn => {
            if (window.currentMaskType === 'same') {
                btn.classList.add('active');
                btn.style.background = '#667eea';
                btn.style.color = 'white';
            } else {
                btn.classList.remove('active');
                btn.style.background = '#e9ecef';
                btn.style.color = '#333';
            }
        });

        differentBtns.forEach(btn => {
            if (window.currentMaskType === 'different') {
                btn.classList.add('active');
                btn.style.background = '#667eea';
                btn.style.color = 'white';
            } else {
                btn.classList.remove('active');
                btn.style.background = '#e9ecef';
                btn.style.color = '#333';
            }
        });

        console.log('✅ 遮罩類型按鈕狀態已更新');
    }, 200);
}

// 🔧 新增：初始化拉桿檢視器
function initializeSliderViewer() {
    const slider = document.getElementById('sliderHandle');
    if (!slider) return;

    const container = slider.parentElement;
    let isDragging = false;

    function startDragging(e) {
        isDragging = true;
        e.preventDefault();
    }

    function drag(e) {
        if (!isDragging) return;

        e.preventDefault();
        const rect = container.getBoundingClientRect();
        const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;

        let position = ((clientX - rect.left) / rect.width) * 100;
        position = Math.max(0, Math.min(100, position));

        updateSliderPosition(position);
    }

    function stopDragging() {
        isDragging = false;
    }

    slider.addEventListener('mousedown', startDragging);
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', stopDragging);

    slider.addEventListener('touchstart', startDragging);
    document.addEventListener('touchmove', drag);
    document.addEventListener('touchend', stopDragging);
}

// 🔧 新增：更新拉桿位置
function updateSliderPosition(position) {
    sliderPosition = position;

    const slider = document.getElementById('sliderHandle');
    const topLayer = document.getElementById('imageLayer2');

    if (slider && topLayer) {
        slider.style.left = `${position}%`;
        topLayer.style.clipPath = `polygon(0 0, ${position}% 0, ${position}% 100%, 0 100%)`;
    }
}

// 🔧 修正：載入原始圖片 - 根據您的需求調整順序
function loadOriginalImages() {
    const layer1 = document.getElementById('imageLayer1');
    const layer2 = document.getElementById('imageLayer2');

    // 🔧 支援歷史模式：使用 window.separatedImages 如果可用
    const currentSeparatedImages = window.separatedImages || separatedImages;

    if (currentSeparatedImages && layer1 && layer2) {
        // 🔧 修正：根據您的期望設定圖片
        // 底層（右側）顯示第二張選擇的圖片（圖片二）
        // 上層（左側，會被拉桿裁切）顯示第一張選擇的圖片（圖片一）
        const rightImagePath = currentSeparatedImages.image2_original.replace(/\\/g, '/');  // 右側顯示圖片二
        const leftImagePath = currentSeparatedImages.image1_original.replace(/\\/g, '/');   // 左側顯示圖片一

        layer1.style.backgroundImage = `url(${API_BASE_URL}/files/${rightImagePath})`; // 底層（右側）圖片二
        layer2.style.backgroundImage = `url(${API_BASE_URL}/files/${leftImagePath})`;  // 上層（左側）圖片一

        console.log('📸 已載入拉桿圖片:');
        console.log('  - 左側（上層，會被裁切）:', leftImagePath, '(圖片一)');
        console.log('  - 右側（底層）:', rightImagePath, '(圖片二)');
    } else {
        console.warn('⚠️ 無法載入原始圖片 - 找不到 separatedImages 資料');
    }
}

// 🔧 新增：分頁切換功能
async function switchViewerTab(tabName) {
    console.log(`🔄 切換檢視器標籤: ${tabName}`);

    // 更新按鈕狀態
    document.querySelectorAll('.viewer-tab').forEach(btn => {
        btn.style.background = 'transparent';
        btn.style.color = '#333';
    });

    // 使用 event.target 如果存在，否則查找對應按鈕
    const activeBtn = event?.target || document.querySelector(`.viewer-tab[onclick*="${tabName}"]`);
    if (activeBtn) {
        activeBtn.style.background = '#667eea';
        activeBtn.style.color = 'white';
    }

    // 切換內容
    document.querySelectorAll('.viewer-content').forEach(content => content.style.display = 'none');

    if (tabName === 'slider') {
        document.getElementById('sliderViewerTab').style.display = 'block';
    } else if (tabName === 'objects') {
        document.getElementById('objectsViewerTab').style.display = 'block';
        // 當切換到物件檢視時，刷新顯示
        console.log('🔄 切換到物件檢視，當前資料:', objectsData);
        await updateObjectDisplay();
    }
}

// 🔧 修改：遮罩控制功能 - 專注於透明度控制
// 🔧 修正版：遮罩開關功能 - 支援歷史模式
function toggleMasks() {
    // 使用正確的變數，根據是否在歷史模式
    if (typeof window.masksVisible !== 'undefined') {
        window.masksVisible = !window.masksVisible;
        var isMasksVisible = window.masksVisible;
    } else {
        masksVisible = !masksVisible;
        var isMasksVisible = masksVisible;
    }

    const toggle = document.getElementById('maskToggle');
    const handle = toggle.querySelector('.toggle-handle');

    console.log(`🎭 Toggle masks: ${isMasksVisible ? 'ON' : 'OFF'}, mask type: ${window.currentMaskType || currentMaskType}`);

    if (isMasksVisible) {
        toggle.style.background = '#667eea';
        handle.style.transform = 'translateX(25px)';

        // 🎯 載入遮罩但保持原圖完全不透明
        loadImagesWithMasks();

        // 🔧 確保原圖層不受透明度控制影響
        const layer1 = document.getElementById('imageLayer1');
        const layer2 = document.getElementById('imageLayer2');

        if (layer1) layer1.style.opacity = '1';
        if (layer2) layer2.style.opacity = '1';

    } else {
        toggle.style.background = '#ddd';
        handle.style.transform = 'translateX(0)';

        // 移除所有遮罩疊加層 - 支援歷史模式
        document.querySelectorAll('.mask-overlay').forEach(overlay => overlay.remove());

        // 在歷史模式下，也需要清除歷史遮罩
        const layer1 = document.getElementById('imageLayer1');
        const layer2 = document.getElementById('imageLayer2');
        if (layer1) removeHistoryMaskOverlays(layer1);
        if (layer2) removeHistoryMaskOverlays(layer2);

        loadOriginalImages();
    }

    console.log(`🎭 遮罩顯示: ${isMasksVisible ? '開啟' : '關閉'}，原圖始終保持完全不透明`);
}


function selectMaskType(type) {
    console.log('🔄 切換遮罩類型:', type);

    // 使用 window 變數確保在歷史模式下正確工作
    if (typeof window.currentMaskType !== 'undefined') {
        window.currentMaskType = type;
    } else {
        currentMaskType = type;
    }

    // 更新按鈕狀態
    document.querySelectorAll('.mask-type-btn').forEach(btn => {
        if (btn.textContent.includes(type === 'same' ? 'Same' : 'Different')) {
            btn.style.background = '#667eea';
            btn.style.color = 'white';
        } else {
            btn.style.background = '#e9ecef';
            btn.style.color = '#333';
        }
    });

    // 🔧 參考正常模式：直接重新載入遮罩（讓 addPngMaskOverlay 處理清除邏輯）
    const isMasksVisible = window.masksVisible !== undefined ? window.masksVisible : masksVisible;
    if (isMasksVisible) {
        loadImagesWithMasks();
    }
}

// 🎨 恢復原始風格：簡潔的透明度控制
function updateOpacity(value) {
    const opacityValue = value / 100;

    // 更新全域變數
    if (typeof window.maskOpacity !== 'undefined') {
        window.maskOpacity = opacityValue;
    } else {
        maskOpacity = opacityValue;
    }

    document.getElementById('opacityValue').textContent = `${value}%`;

    // 更新所有類型的遮罩疊加層的透明度
    const maskOverlaySelectors = [
        '.mask-overlay',
        '.history-mask-overlay'
    ];

    maskOverlaySelectors.forEach(selector => {
        const overlays = document.querySelectorAll(selector);
        overlays.forEach(overlay => {
            overlay.style.opacity = opacityValue;
        });
    });

    console.log(`🎭 遮罩透明度更新: ${value}%`);
}

// 🎨 清爽版：載入遮罩圖片
function loadImagesWithMasks() {
    // 🔧 支援歷史模式：使用 window.separatedImages 如果可用
    const currentSeparatedImages = window.separatedImages || separatedImages;

    if (!currentSeparatedImages) {
        console.warn('⚠️ 沒有可用的分離圖片資料');
        return;
    }

    console.log('🎭 載入遮罩，使用資料:', currentSeparatedImages);

    const layer1 = document.getElementById('imageLayer1');
    const layer2 = document.getElementById('imageLayer2');

    const rightImagePath = currentSeparatedImages.image2_original.replace(/\\/g, '/');
    const leftImagePath = currentSeparatedImages.image1_original.replace(/\\/g, '/');

    // 🔧 使用正確的遮罩類型變數
    const currentMaskTypeValue = window.currentMaskType || currentMaskType;

    if (currentMaskTypeValue === 'same') {
        const leftMaskPath = currentSeparatedImages.image1_same_masks.replace(/\\/g, '/');
        const rightMaskPath = currentSeparatedImages.image2_same_masks.replace(/\\/g, '/');

        addPngMaskOverlay(layer1, rightImagePath, rightMaskPath, 'same-mask-2');
        addPngMaskOverlay(layer2, leftImagePath, leftMaskPath, 'same-mask-1');

    } else {
        const disappearedPath = currentSeparatedImages.image1_disappeared_masks.replace(/\\/g, '/');
        const appearedPath = currentSeparatedImages.image2_appeared_masks.replace(/\\/g, '/');

        addPngMaskOverlay(layer1, rightImagePath, appearedPath, 'appeared-mask');
        addPngMaskOverlay(layer2, leftImagePath, disappearedPath, 'disappeared-mask');
    }

    console.log(`✅ 遮罩載入完成，類型: ${currentMaskTypeValue}`);
}


// 🎨 簡化版：回歸原始清爽風格
function addPngMaskOverlay(targetElement, backgroundPath, maskPath, maskId) {
    // 移除現有的遮罩疊加層
    const existingOverlay = targetElement.querySelector('.mask-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }

    // 設定背景圖片（原始圖片）
    targetElement.style.backgroundImage = `url(${API_BASE_URL}/files/${backgroundPath})`;
    targetElement.style.opacity = '1';

    // 🔧 使用正確的透明度變數
    const currentOpacity = window.maskOpacity !== undefined ? window.maskOpacity : maskOpacity;

    // 創建簡潔的遮罩疊加層
    const overlay = document.createElement('div');
    overlay.className = 'mask-overlay';
    overlay.id = maskId;
    overlay.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url(${API_BASE_URL}/files/${maskPath});
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        opacity: ${currentOpacity};
        pointer-events: none;
        z-index: 5;
        transition: opacity 0.2s ease;
    `;

    targetElement.appendChild(overlay);

    if (getComputedStyle(targetElement).position === 'static') {
        targetElement.style.position = 'relative';
    }

    console.log(`✅ 遮罩載入: ${maskPath.split('/').pop()}, 透明度: ${currentOpacity}`);
}

// 🔧 新增：物件名稱翻譯函數
function translateObjectName(name) {
    if (!name) return name;

    // 翻譯中文物件名稱為英文
    return name
        .replace(/消失物件\s*(\d+)/g, 'Disappeared Object $1')
        .replace(/新增物件\s*(\d+)/g, 'Appeared Object $1')
        .replace(/消失物件/g, 'Disappeared Object')
        .replace(/新增物件/g, 'Appeared Object');
}

// 🔧 新增：物件檢視功能
async function switchObjectType(type) {
    currentObjectType = type;
    currentObjectIndex = 0;

    // 更新按鈕狀態
    document.querySelectorAll('.object-tab').forEach(btn => {
        if (btn.textContent.includes(type === 'disappeared' ? '消失' : '新增')) {
            btn.style.background = '#667eea';
            btn.style.color = 'white';
        } else {
            btn.style.background = 'transparent';
            btn.style.color = '#333';
        }
    });

    await updateObjectDisplay();
}

async function previousObject() {
    if (currentObjectIndex > 0) {
        currentObjectIndex--;
        await updateObjectDisplay();
    }
}

async function nextObject() {
    const objects = objectsData[currentObjectType];
    if (currentObjectIndex < objects.length - 1) {
        currentObjectIndex++;
        await updateObjectDisplay();
    }
}

async function updateObjectDisplay() {
    const objects = objectsData[currentObjectType];
    const display = document.getElementById('objectDisplay');
    const info = document.getElementById('objectInfo');
    const prevBtn = document.getElementById('prevObjectBtn');
    const nextBtn = document.getElementById('nextObjectBtn');

    if (!display || !info || !prevBtn || !nextBtn) {
        console.warn('⚠️ 物件檢視元素未找到');
        return;
    }

    console.log(`🔍 更新物件顯示: ${currentObjectType}, 索引: ${currentObjectIndex}, 總數: ${objects.length}`);

    // 更新物件資訊
    info.textContent = `${currentObjectIndex + 1} / ${objects.length}`;

    // 更新按鈕狀態
    prevBtn.disabled = currentObjectIndex === 0 || objects.length === 0;
    nextBtn.disabled = currentObjectIndex === objects.length - 1 || objects.length === 0;

    prevBtn.style.opacity = prevBtn.disabled ? '0.5' : '1';
    nextBtn.style.opacity = nextBtn.disabled ? '0.5' : '1';

    // 顯示物件圖片
    if (objects.length > 0 && objects[currentObjectIndex]) {
        const currentObject = objects[currentObjectIndex];
        console.log('🖼️ 顯示物件:', currentObject);

        // 構建正確的檔案路徑 (局部裁切圖像)
        const beforePath = currentObject.before_path.replace(/\\/g, '/');
        const afterPath = currentObject.after_path.replace(/\\/g, '/');
        const maskPath = currentObject.mask_path ? currentObject.mask_path.replace(/\\/g, '/') : '';

        // 清理路徑
        const cleanBeforePath = beforePath.startsWith('/') ? beforePath.substring(1) : beforePath;
        const cleanAfterPath = afterPath.startsWith('/') ? afterPath.substring(1) : afterPath;
        const cleanMaskPath = maskPath.startsWith('/') ? maskPath.substring(1) : maskPath;

        console.log('📁 Before路徑:', cleanBeforePath);
        console.log('📁 After路徑:', cleanAfterPath);
        console.log('� Mask路徑:', cleanMaskPath);
        console.log('�🔢 目前運行編號:', window.currentRunNumber);
        console.log('🔗 API_BASE_URL:', API_BASE_URL);

        // 動態獲取當前運行編號
        let runNumber = String(window.currentRunNumber || '').padStart(3, '0');

        // 如果沒有運行編號，嘗試從API獲取
        if (!runNumber || runNumber === '000') {
            try {
                const response = await fetch(`${API_BASE_URL}/current_run`);
                if (response.ok) {
                    const result = await response.json();
                    if (result.status === 'success' && result.data.run_number) {
                        runNumber = String(result.data.run_number).padStart(3, '0');
                        window.currentRunNumber = result.data.run_number;
                        console.log('📥 從API獲取運行編號:', runNumber);
                    } else {
                        console.warn('⚠️ 無法獲取當前運行編號，使用預設值');
                        runNumber = '048'; // 根據您提到的當前運行
                    }
                } else {
                    console.warn('⚠️ 獲取運行編號API失敗，使用預設值');
                    runNumber = '048'; // 根據您提到的當前運行
                }
            } catch (error) {
                console.warn('⚠️ 獲取運行編號時發生錯誤，使用預設值:', error);
                runNumber = '048'; // 根據您提到的當前運行
            }
        }

        const beforeImageUrl = `${API_BASE_URL}/files/results/runs/run_${runNumber}/detection/${cleanBeforePath}`;
        const afterImageUrl = `${API_BASE_URL}/files/results/runs/run_${runNumber}/detection/${cleanAfterPath}`;
        const maskImageUrl = cleanMaskPath ? `${API_BASE_URL}/files/results/runs/run_${runNumber}/detection/${cleanMaskPath}` : '';

        console.log('🖼️ Before圖片URL:', beforeImageUrl);
        console.log('🖼️ After圖片URL:', afterImageUrl);
        console.log('🖼️ Mask圖片URL:', maskImageUrl);

        display.innerHTML = `
            <!-- 🎨 改進版：現代化物件檢視器 - 水平布局 -->
            <div style="display: flex; width: 100%; height: 300px; gap: 15px; background: #f8f9fa; border-radius: 12px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">

                <!-- 圖片區域 - 佔左側較大空間 -->
                <div style="flex: 3; display: flex; gap: 15px; height: 100%;">
                    <!-- Before Change Image -->
                    <div class="image-container" style="width: 50%; height: 100%; position: relative; overflow: hidden; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <img id="beforeImage" src="${beforeImageUrl}"
                             style="width: 100%; height: 100%; object-fit: contain; object-position: center; background: white;"
                             alt="Before Change - Local Area"
                             onload="console.log('Before image loaded successfully')"
                             onerror="console.error('Failed to load Before image:', this.src)">
                        ${maskImageUrl ? `
                        <canvas id="beforeMaskCanvas"
                                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: none; opacity: 0.7; z-index: 3; pointer-events: none;"
                                alt="Before Change Mask Overlay">
                        </canvas>` : ''}
                        <div style="position: absolute; top: 12px; left: 12px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);">
                            📷 Before Change
                        </div>
                    </div>

                    <!-- Separator Line -->
                    <div style="width: 2px; background: linear-gradient(to bottom, #667eea, #764ba2); border-radius: 1px; opacity: 0.6;"></div>

                    <!-- After Change Image -->
                    <div class="image-container" style="width: 50%; height: 100%; position: relative; overflow: hidden; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <img id="afterImage" src="${afterImageUrl}"
                             style="width: 100%; height: 100%; object-fit: contain; object-position: center; background: white;"
                             alt="After Change - Local Area"
                             onload="console.log('After image loaded successfully')"
                             onerror="console.error('Failed to load After image:', this.src)">
                        ${maskImageUrl ? `
                        <canvas id="afterMaskCanvas"
                                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: none; opacity: 0.7; z-index: 3; pointer-events: none;"
                                alt="After Change Mask Overlay">
                        </canvas>` : ''}
                        <div style="position: absolute; top: 12px; left: 12px; background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);">
                            📷 After Change
                        </div>
                    </div>
                </div>

                <!-- 資訊區域 - 佔右側空間 -->
                <div style="flex: 1; display: flex; flex-direction: column; height: 100%;">
                    <!-- 物件標題 -->
                    <div style="background: white; border-radius: 12px; padding: 15px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                            <div style="width: 6px; height: 6px; background: ${currentObjectType === 'disappeared' ? '#dc3545' : '#28a745'}; border-radius: 50%; box-shadow: 0 0 8px ${currentObjectType === 'disappeared' ? 'rgba(220, 53, 69, 0.4)' : 'rgba(40, 167, 69, 0.4)'};"></div>
                            <h4 style="margin: 0; font-size: 14px; font-weight: 700; color: #2c3e50;">
                                ${translateObjectName(currentObject.name)}
                            </h4>
                        </div>
                    </div>

                    <!-- 統計數據 - 垂直排列，緊湊設計 -->
                    <div style="flex: 1; display: flex; flex-direction: column; gap: 8px;">
                        <!-- Change Magnitude Card -->
                        <div class="stat-card" style="background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 8px; padding: 12px; color: white; text-align: center; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2); flex: 1; display: flex; flex-direction: column; justify-content: center;">
                            <div style="font-size: 18px; font-weight: 800; line-height: 1; margin-bottom: 2px; text-shadow: 0 1px 3px rgba(0,0,0,0.2);">
                                ${currentObject.changeRatio || currentObject.change_ratio || 'N/A'}${currentObject.changeRatio || currentObject.change_ratio ? '%' : ''}
                            </div>
                            <div style="font-size: 9px; opacity: 0.95; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                                Change Magnitude
                            </div>
                        </div>

                        <!-- Confidence Card -->
                        <div class="stat-card" style="background: linear-gradient(135deg, #28a745, #20c997); border-radius: 8px; padding: 12px; color: white; text-align: center; box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2); flex: 1; display: flex; flex-direction: column; justify-content: center;">
                            <div style="font-size: 18px; font-weight: 800; line-height: 1; margin-bottom: 2px; text-shadow: 0 1px 3px rgba(0,0,0,0.2);">
                                ${currentObject.confidence || currentObject.score || 'N/A'}${currentObject.confidence || currentObject.score ? '%' : ''}
                            </div>
                            <div style="font-size: 9px; opacity: 0.95; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                                Confidence
                            </div>
                        </div>

                        <!-- Area Size Card -->
                        ${currentObject.bbox ? `
                        <div class="stat-card" style="background: linear-gradient(135deg, #6c757d, #495057); border-radius: 8px; padding: 12px; color: white; text-align: center; box-shadow: 0 2px 8px rgba(108, 117, 125, 0.2); flex: 1; display: flex; flex-direction: column; justify-content: center;">
                            <div style="font-size: 16px; font-weight: 700; line-height: 1; margin-bottom: 2px; text-shadow: 0 1px 3px rgba(0,0,0,0.2);">
                                ${currentObject.bbox.width || currentObject.width || 'N/A'}×${currentObject.bbox.height || currentObject.height || 'N/A'}
                            </div>
                            <div style="font-size: 9px; opacity: 0.95; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                                Area Size
                            </div>
                        </div>` : `
                        <div class="stat-card" style="background: linear-gradient(135deg, #6c757d, #495057); border-radius: 8px; padding: 12px; color: white; text-align: center; box-shadow: 0 2px 8px rgba(108, 117, 125, 0.2); flex: 1; display: flex; flex-direction: column; justify-content: center;">
                            <div style="font-size: 16px; font-weight: 700; line-height: 1; margin-bottom: 2px; text-shadow: 0 1px 3px rgba(0,0,0,0.2);">
                                ${currentObject.width || 120}×${currentObject.height || 100}
                            </div>
                            <div style="font-size: 9px; opacity: 0.95; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                                Area Size
                            </div>
                        </div>`}
                    </div>
                </div>
            </div>
        `;
    } else {
        display.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #666; flex-direction: column;">
                <div style="font-size: 48px; margin-bottom: 16px;">📷</div>
                <div style="font-size: 16px; margin-bottom: 8px;">No ${currentObjectType === 'disappeared' ? 'disappeared' : 'appeared'} objects</div>
                <div style="font-size: 14px;">請執行檢測以產生物件變化結果</div>
            </div>
        `;
    }

    // 恢復checkbox狀態和遮罩顯示
    setTimeout(() => {
        const maskToggle = document.getElementById('objectMaskToggle');
        if (maskToggle) {
            maskToggle.checked = objectMaskVisible;

            // 🔧 同時更新切換開關的視覺狀態
            const toggleSwitch = maskToggle.parentNode.querySelector('.toggle-switch');
            const toggleHandle = toggleSwitch ? toggleSwitch.querySelector('.toggle-handle') : null;

            if (toggleSwitch && toggleHandle) {
                if (objectMaskVisible) {
                    // 開啟狀態：藍紫色背景，手柄向右
                    toggleSwitch.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
                    toggleHandle.style.transform = 'translateX(24px)';
                    toggleHandle.style.boxShadow = '0 2px 8px rgba(102, 126, 234, 0.4)';
                } else {
                    // 關閉狀態：灰色背景，手柄向左
                    toggleSwitch.style.background = '#ddd';
                    toggleHandle.style.transform = 'translateX(0)';
                    toggleHandle.style.boxShadow = '0 2px 6px rgba(0,0,0,0.2)';
                }
            }

            // 如果遮罩應該顯示，則自動觸發顯示
            if (objectMaskVisible) {
                const beforeMaskCanvas = document.getElementById('beforeMaskCanvas');
                const afterMaskCanvas = document.getElementById('afterMaskCanvas');

                if (beforeMaskCanvas && afterMaskCanvas) {
                    loadColoredMask();
                    beforeMaskCanvas.style.display = 'block';
                    afterMaskCanvas.style.display = 'block';
                    console.log('✅ 自動恢復遮罩顯示狀態');
                }
            }
        }
    }, 100); // 短暫延遲確保DOM元素已創建
}

// 新增：切換遮罩顯示功能 - 支援彩色遮罩
function toggleMaskDisplay() {
    const beforeMaskCanvas = document.getElementById('beforeMaskCanvas');
    const afterMaskCanvas = document.getElementById('afterMaskCanvas');
    const maskToggle = document.getElementById('objectMaskToggle');

    // 🔧 新增：獲取切換開關容器元素
    const toggleSwitch = maskToggle ? maskToggle.parentNode.querySelector('.toggle-switch') : null;
    const toggleHandle = toggleSwitch ? toggleSwitch.querySelector('.toggle-handle') : null;

    console.log('🎭 調試 - 找到的元素:', {
        beforeMaskCanvas: !!beforeMaskCanvas,
        afterMaskCanvas: !!afterMaskCanvas,
        maskToggle: !!maskToggle,
        toggleSwitch: !!toggleSwitch,
        toggleHandle: !!toggleHandle
    });

    if (!beforeMaskCanvas || !afterMaskCanvas || !maskToggle) {
        console.warn('⚠️ 遮罩元素未找到');
        return;
    }

    const isChecked = maskToggle.checked;
    console.log('🎭 調試 - checkbox狀態:', isChecked);

    // 🔧 更新切換開關的視覺狀態
    if (toggleSwitch && toggleHandle) {
        if (isChecked) {
            // 開啟狀態：藍紫色背景，手柄向右
            toggleSwitch.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
            toggleHandle.style.transform = 'translateX(24px)';
            toggleHandle.style.boxShadow = '0 2px 8px rgba(102, 126, 234, 0.4)';
        } else {
            // 關閉狀態：灰色背景，手柄向左
            toggleSwitch.style.background = '#ddd';
            toggleHandle.style.transform = 'translateX(0)';
            toggleHandle.style.boxShadow = '0 2px 6px rgba(0,0,0,0.2)';
        }
    }

    // 更新全局狀態
    objectMaskVisible = isChecked;

    if (isChecked) {
        // 顯示遮罩並載入彩色遮罩
        console.log('🎭 準備顯示遮罩...');
        loadColoredMask();
        beforeMaskCanvas.style.display = 'block';
        afterMaskCanvas.style.display = 'block';
        console.log('✅ 彩色遮罩疊加已顯示');
    } else {
        console.log('🎭 準備隱藏遮罩...');
        beforeMaskCanvas.style.display = 'none';
        afterMaskCanvas.style.display = 'none';
        console.log('❌ 遮罩疊加已隱藏');
    }
}

// 新增：載入彩色遮罩函數
function loadColoredMask() {
    const currentObjects = objectsData[currentObjectType];
    if (!currentObjects || currentObjects.length === 0) {
        console.warn('⚠️ 沒有可用的物件資料');
        return;
    }

    const currentObject = currentObjects[currentObjectIndex];
    if (!currentObject) {
        console.warn('⚠️ 沒有找到當前物件');
        return;
    }

    const maskPath = currentObject.mask_path;

    if (!maskPath) {
        console.warn('⚠️ 當前物件沒有遮罩路徑');
        return;
    }

    console.log('🎭 載入遮罩:', maskPath, '類型:', currentObjectType);

    // 確定遮罩顏色
    const maskColor = currentObjectType === 'disappeared' ?
        { r: 255, g: 0, b: 0 } :    // 紅色 - 消失
        { r: 0, g: 255, b: 0 };     // 綠色 - 新增

    // 載入並處理遮罩
    const maskImage = new Image();
    maskImage.crossOrigin = 'anonymous';

    maskImage.onload = function() {
        drawColoredMask('beforeMaskCanvas', this, maskColor);
        drawColoredMask('afterMaskCanvas', this, maskColor);
        console.log(`✅ 載入${currentObjectType === 'disappeared' ? '紅色消失' : '綠色新增'}遮罩成功`);
    };

    maskImage.onerror = function() {
        console.error('❌ 載入遮罩圖片失敗:', maskPath);
    };

    // 構建遮罩圖片URL - 使用動態運行編號
    const cleanMaskPath = maskPath.startsWith('/') ? maskPath.substring(1) : maskPath;
    let runNumber = String(window.currentRunNumber || '048').padStart(3, '0');
    const maskImageUrl = `${API_BASE_URL}/files/results/runs/run_${runNumber}/detection/${cleanMaskPath}`;

    console.log('🔗 遮罩圖片URL:', maskImageUrl);
    maskImage.src = maskImageUrl;
}

// 新增：繪製彩色遮罩
function drawColoredMask(canvasId, maskImage, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.warn('⚠️ Canvas元素未找到:', canvasId);
        return;
    }

    const ctx = canvas.getContext('2d');
    const container = canvas.parentElement;

    // 獲取對應的圖片元素來計算正確的尺寸
    const imageId = canvasId.includes('before') ? 'beforeImage' : 'afterImage';
    const img = document.getElementById(imageId);

    if (!img) {
        console.warn('⚠️ 對應的圖片元素未找到:', imageId);
        return;
    }

    // 設置canvas的實際尺寸為容器尺寸
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    // 清除canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 計算圖片在容器中的實際顯示區域 (object-fit: contain 的效果)
    const containerAspect = container.clientWidth / container.clientHeight;
    const imageAspect = img.naturalWidth / img.naturalHeight;

    let displayWidth, displayHeight, offsetX, offsetY;

    if (containerAspect > imageAspect) {
        // 容器比圖片寬，圖片會垂直填滿，左右留白
        displayHeight = container.clientHeight;
        displayWidth = displayHeight * imageAspect;
        offsetX = (container.clientWidth - displayWidth) / 2;
        offsetY = 0;
    } else {
        // 容器比圖片高，圖片會水平填滿，上下留白
        displayWidth = container.clientWidth;
        displayHeight = displayWidth / imageAspect;
        offsetX = 0;
        offsetY = (container.clientHeight - displayHeight) / 2;
    }

    // 創建臨時canvas來處理遮罩
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = maskImage.width;
    tempCanvas.height = maskImage.height;

    // 繪製原始遮罩到臨時canvas
    tempCtx.drawImage(maskImage, 0, 0);

    // 獲取圖像數據
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const data = imageData.data;

    // 將白色區域替換為指定顏色
    for (let i = 0; i < data.length; i += 4) {
        const alpha = data[i + 3];
        if (alpha > 128) { // 如果像素不透明
            data[i] = color.r;     // 紅色分量
            data[i + 1] = color.g; // 綠色分量
            data[i + 2] = color.b; // 藍色分量
            data[i + 3] = 180;     // 透明度 (70% 不透明)
        }
    }

    // 將處理後的數據放回
    tempCtx.putImageData(imageData, 0, 0);

    // 繪製遮罩到正確的位置和尺寸
    ctx.drawImage(tempCanvas, offsetX, offsetY, displayWidth, displayHeight);

    console.log(`✅ 遮罩繪製完成: ${canvasId}, 尺寸: ${displayWidth.toFixed(0)}x${displayHeight.toFixed(0)}, 偏移: ${offsetX.toFixed(0)},${offsetY.toFixed(0)}`);
}

// 🔧 修復版：在網頁中顯示結果圖片
// 修改版：直接顯示檢測結果數據，不載入圖片
function displayResultImages(data) {
    const container = document.getElementById('resultImageContainer');
    if (!container) return;

    console.log('🔍 準備顯示優化後的檢測結果:', data);

    let resultHTML = '';

    // 檢測結果摘要
    const summary = data.analysis_summary || {};
    const confirmedDisappeared = summary.confirmed_disappeared || 0;
    const confirmedAppeared = summary.confirmed_appeared || 0;
    const totalChanges = summary.total_confirmed_changes || 0;

    // 構建簡化的結果顯示
    resultHTML = `
        <div class="detection-results-container">
            <!-- 視覺化結果展示區域 -->
            <div class="visualization-results">
                <h4>📊 檢測結果視覺化</h4>
                <div class="image-grid" id="visualizationImageGrid">
                    <!-- 圖片將通過 JavaScript 動態載入 -->
                </div>
            </div>
        </div>
    `;

    container.innerHTML = resultHTML;

    // 🔧 載入指定的視覺化圖片
    loadVisualizationImages(data);
}

// 載入視覺化圖片函式 - 新增縮放功能
function loadVisualizationImages(data) {
    const imageGrid = document.getElementById('visualizationImageGrid');
    if (!imageGrid) return;

    const imagesToLoad = [];

    // 1. 遮罩匹配結果（保留）
    if (window.maskMatchingOutputDir) {
        imagesToLoad.push({
            path: `${window.maskMatchingOutputDir}/optimized_mask_matching_results.jpg`,
            title: '遮罩匹配結果',
            description: '物件匹配和分類結果'
        });
    }

    // 2. 詳細變化對比 - 消失遮罩
    if (data.output_directory && data.analysis_summary.confirmed_disappeared > 0) {
        imagesToLoad.push({
            path: `${data.output_directory}/disappeared_masks_comparison.jpg`,
            title: 'Disappeared Objects Detailed Comparison',
            description: `Show the top 5 most obvious disappeared objects`
        });
    }

    // 3. 詳細變化對比 - 新增遮罩
    if (data.output_directory && data.analysis_summary.confirmed_appeared > 0) {
        imagesToLoad.push({
            path: `${data.output_directory}/appeared_masks_comparison.jpg`,
            title: 'Appeared Objects Detailed Comparison',
            description: `Shows the top 5 most obvious appeared objects`
        });
    }

    // 動態載入圖片
    imagesToLoad.forEach((imgInfo, index) => {
        const imageContainer = document.createElement('div');
        imageContainer.className = 'image-item';

        const titleElement = document.createElement('h5');
        titleElement.textContent = imgInfo.title;
        titleElement.className = 'image-title';

        const descElement = document.createElement('p');
        descElement.textContent = imgInfo.description;
        descElement.className = 'image-description';

        // 🔧 新增：圖片控制區域
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'image-controls';

        const zoomInBtn = document.createElement('button');
        zoomInBtn.textContent = '🔍 放大';
        zoomInBtn.className = 'zoom-btn zoom-in';

        const zoomOutBtn = document.createElement('button');
        zoomOutBtn.textContent = '🔍 縮小';
        zoomOutBtn.className = 'zoom-btn zoom-out';

        const fullscreenBtn = document.createElement('button');
        fullscreenBtn.textContent = '🖼️ 全螢幕';
        fullscreenBtn.className = 'zoom-btn fullscreen';

        const imgElement = document.createElement('img');
        imgElement.src = `${API_BASE_URL}/files/${imgInfo.path}`;
        imgElement.alt = imgInfo.title;
        imgElement.className = 'result-image';
        imgElement.dataset.scale = '1';

        // 🔧 新增：縮放功能
        let currentScale = 1;

        zoomInBtn.onclick = function() {
            currentScale = Math.min(currentScale * 1.2, 3);
            imgElement.style.transform = `scale(${currentScale})`;
            imgElement.style.transformOrigin = 'center';
            imgElement.dataset.scale = currentScale;
        };

        zoomOutBtn.onclick = function() {
            currentScale = Math.max(currentScale / 1.2, 0.5);
            imgElement.style.transform = `scale(${currentScale})`;
            imgElement.dataset.scale = currentScale;
        };

        fullscreenBtn.onclick = function() {
            openImageModal(imgElement.src, imgInfo.title);
        };

        // 圖片載入事件
        imgElement.onload = function() {
            console.log(`✅ 圖片載入成功: ${imgInfo.title}`);
        };

        imgElement.onerror = function() {
            console.log(`❌ 圖片載入失敗: ${imgInfo.path}`);
            imgElement.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjBmMGYwIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuWclueLh+eEoeazleizieWFpTwvdGV4dD48L3N2Zz4=';
        };

        controlsDiv.appendChild(zoomInBtn);
        controlsDiv.appendChild(zoomOutBtn);
        controlsDiv.appendChild(fullscreenBtn);

        imageContainer.appendChild(titleElement);
        imageContainer.appendChild(descElement);
        imageContainer.appendChild(controlsDiv);
        imageContainer.appendChild(imgElement);
        imageGrid.appendChild(imageContainer);
    });
}

// 🔧 新增：全螢幕模態視窗
function openImageModal(imageSrc, title) {
    // 創建模態視窗
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>${title}</h3>
                <button class="close-btn" onclick="this.parentElement.parentElement.parentElement.remove()">✕</button>
            </div>
            <div class="modal-body">
                <img src="${imageSrc}" alt="${title}" style="max-width: 95vw; max-height: 85vh; object-fit: contain;">
            </div>
        </div>
    `;

    // 點擊背景關閉
    modal.onclick = function(e) {
        if (e.target === modal) {
            modal.remove();
        }
    };

    document.body.appendChild(modal);
}

// 🔧 新增：生成視覺化圖片的函式
function generateVisualizationImages(data) {
    const imagePaths = [];

    // 🎯 優先顯示四面板遮罩視覺化
    if (data.four_panel_visualization_path) {
        imagePaths.push({
            path: data.four_panel_visualization_path,
            title: '四面板遮罩視覺化',
            description: '分類展示共有、消失和新增的遮罩',
            priority: 1
        });
    }

    // 原有的檢測結果圖片
    if (data.visualization_path) {
        imagePaths.push({
            path: data.visualization_path,
            title: '變化檢測結果',
            description: '標示確認變化的物件',
            priority: 2
        });
    }

    // 遮罩匹配結果
    if (window.maskMatchingOutputDir) {
        imagePaths.push({
            path: `${window.maskMatchingOutputDir}/optimized_mask_matching_results.jpg`,
            title: '遮罩匹配結果',
            description: '物件匹配和分類結果',
            priority: 3
        });
    }

    // 詳細對比圖
    if (data.detailed_comparison_path) {
        imagePaths.push({
            path: data.detailed_comparison_path,
            title: '詳細變化對比',
            description: '重點物件變化詳細展示',
            priority: 4
        });
    }

    // 按優先級排序
    imagePaths.sort((a, b) => a.priority - b.priority);

    if (imagePaths.length === 0) {
        return '<p class="no-images-message">沒有可用的視覺化圖片</p>';
    }

    let visualizationHTML = '';

    imagePaths.forEach((imgInfo, index) => {
        // 四面板視覺化使用更大的顯示尺寸
        const isMainVisualization = imgInfo.priority === 1;
        const containerClass = isMainVisualization ? 'main-visualization' : 'secondary-visualization';

        visualizationHTML += `
            <div class="visualization-item ${containerClass}">
                <h5 class="visualization-title">${imgInfo.title}</h5>
                <p class="visualization-description">${imgInfo.description}</p>
                <div class="image-container">
                    <img
                        src="${API_BASE_URL}/files/${imgInfo.path}"
                        alt="${imgInfo.title}"
                        class="visualization-image"
                        loading="lazy"
                        onload="this.classList.add('loaded')"
                        onerror="this.classList.add('error'); this.alt='圖片載入失敗';"
                    />
                    <div class="image-loading">Loading...</div>
                </div>
            </div>
        `;
    });

    return visualizationHTML;
}

// 生成詳細檢測結果
function generateDetailedResults(results) {
    if (!results || Object.keys(results).length === 0) {
        return '<div class="no-details">No detailed results to display</div>';
    }

    let detailsHTML = '<div class="result-card details-card"><h4>📋 Detailed Detection Results</h4>';

    // 分類處理結果
    const disappearResults = [];
    const appearResults = [];

    Object.entries(results).forEach(([maskName, result]) => {
        const resultInfo = {
            name: maskName,
            status: result.status,
            confidence: (result.confidence * 100).toFixed(1),
            changeRatio: (result.change_ratio * 100).toFixed(1),
            changedPixels: result.changed_pixels,
            maskArea: result.mask_area
        };

        if (result.category === 'disappear_analysis' && result.status === 'confirmed_disappeared') {
            disappearResults.push(resultInfo);
        } else if (result.category === 'newadded_analysis' && result.status === 'confirmed_appeared') {
            appearResults.push(resultInfo);
        }
    });

    // Disappeared objects results
    if (disappearResults.length > 0) {
        detailsHTML += `
            <div class="category-results disappeared-results">
                <h5>📤 Confirm Disappeared Objects</h5>
                <div class="results-list">
        `;

        disappearResults.forEach(item => {
            detailsHTML += `
                <div class="result-item disappeared-item">
                    <div class="item-header">
                        <span class="item-name">${translateObjectName(item.name)}</span>
                        <span class="confidence-badge">${item.confidence}%</span>
                    </div>
                    <div class="item-details">
                        <span>Change Ratio: ${item.changeRatio}%</span>
                        <span>Changed Pixels: ${item.changedPixels}</span>
                        <span>Mask Area: ${item.maskArea} pixels</span>
                    </div>
                </div>
            `;
        });

        detailsHTML += '</div></div>';
    }

    // Appeared objects results
    if (appearResults.length > 0) {
        detailsHTML += `
            <div class="category-results appeared-results">
                <h5>📥 Confirm Appeared Objects</h5>
                <div class="results-list">
        `;

        appearResults.forEach(item => {
            detailsHTML += `
                <div class="result-item appeared-item">
                    <div class="item-header">
                        <span class="item-name">${translateObjectName(item.name)}</span>
                        <span class="confidence-badge">${item.confidence}%</span>
                    </div>
                    <div class="item-details">
                        <span>Change Ratio: ${item.changeRatio}%</span>
                        <span>Changed Pixels: ${item.changedPixels}</span>
                        <span>Mask Area: ${item.maskArea} pixels</span>
                    </div>
                </div>
            `;
        });

        detailsHTML += '</div></div>';
    }

    // 🔧 新增：如果沒有找到任何確認的變化，顯示相應訊息
    if (disappearResults.length === 0 && appearResults.length === 0) {
        detailsHTML += `
            <div class="no-confirmed-changes">
                <div class="info-message">
                    <h5>ℹ️ 檢測結果</h5>
                    <p>The AI system has completed detailed analysis but found no significant changes requiring confirmation.</p>
                    <p>這可能表示：</p>
                    <ul>
                        <li>兩張圖片之間沒有實質性的物件變化</li>
                        <li>檢測到的變化可能是光線、陰影或微小移動造成的</li>
                        <li>系統的過濾機制成功排除了誤判</li>
                    </ul>
                </div>
            </div>
        `;
    }

    // 🔧 新增：結果摘要統計
    const totalConfirmedChanges = disappearResults.length + appearResults.length;
    // 檢測摘要統計已移除

    detailsHTML += '</div>';
    return detailsHTML;
}

// 🧪 Debug function - 在瀏覽器控制台中使用
window.debugParameters = function() {
    console.log('=== 🔍 參數調試資訊 ===');
    console.log('1. 當前analysisParameters物件:');
    console.log(analysisParameters);

    console.log('2. HTML表單中的參數值:');
    Object.keys(analysisParameters).forEach(key => {
        const element = document.getElementById(key);
        if (element) {
            console.log(`  - ${key}: HTML="${element.value || element.checked}" vs Memory="${analysisParameters[key]}"`);
        }
    });

    console.log('3. localStorage中的參數:');
    try {
        const stored = localStorage.getItem('analysisParameters');
        console.log(stored ? JSON.parse(stored) : 'No stored parameters');
    } catch (e) {
        console.log('Error reading localStorage:', e);
    }

    console.log('=== 調試完成 ===');
};

// ===== Analysis Parameters Management =====
const analysisParameters = {
    // Sky Mask Removal
    enableSkyRemoval: true,

    // SAM2 Segmentation Parameters (Using previous working parameters)
    pointsPerSide: 48,
    pointsPerBatch: 64,
    predIouThresh: 0.75,
    stabilityScoreThresh: 0.9,
    stabilityScoreOffset: 1.0,
    minMaskRegionArea: 10000,

    // Mask Matching Parameters
    iouThreshold: 0.3,
    distanceThreshold: 20,
    similarityThreshold: 0.35,

    // Additional parameters that might be used
    erosionKernelSize: 5,
    dilationKernelSize: 5
};

// Load parameters from localStorage on page load
function loadParametersFromStorage() {
    const saved = localStorage.getItem('analysisParameters');
    if (saved) {
        const savedParams = JSON.parse(saved);
        Object.assign(analysisParameters, savedParams);
    }
    updateParameterUI();
}

// Update UI elements with current parameter values
function updateParameterUI() {
    // Update all input elements with current parameter values
    Object.keys(analysisParameters).forEach(key => {
        const element = document.getElementById(key);
        if (element) {
            if (element.type === 'checkbox') {
                element.checked = analysisParameters[key];
                // Update toggle text
                const toggleText = element.closest('.toggle-switch-container')?.querySelector('.toggle-text');
                if (toggleText) {
                    toggleText.textContent = element.checked ? 'Enabled' : 'Disabled';
                }
            } else {
                element.value = analysisParameters[key];
            }
        }
    });
}

// Save parameters to localStorage and update internal object
function saveParameters() {
    console.log('📋 保存參數：從HTML表單更新內部物件...');

    // Update internal object from UI
    Object.keys(analysisParameters).forEach(key => {
        const element = document.getElementById(key);
        if (element) {
            const oldValue = analysisParameters[key];
            if (element.type === 'checkbox') {
                analysisParameters[key] = element.checked;
            } else {
                analysisParameters[key] = parseFloat(element.value) || element.value;
            }
            console.log(`  - ${key}: ${oldValue} → ${analysisParameters[key]}`);
        }
    });

    // Save to localStorage
    localStorage.setItem('analysisParameters', JSON.stringify(analysisParameters));
    console.log('✅ 參數已保存到localStorage');

    // Show confirmation
    showNotification('Parameters saved successfully!', 'success');
}

// Load parameters from localStorage and update UI
function loadParametersFromStorage() {
    console.log('📋 從localStorage載入參數...');

    try {
        const storedParams = localStorage.getItem('analysisParameters');
        if (storedParams) {
            const parsedParams = JSON.parse(storedParams);

            // Update internal object
            Object.keys(parsedParams).forEach(key => {
                if (key in analysisParameters) {
                    analysisParameters[key] = parsedParams[key];
                }
            });

            console.log('✅ 載入的參數:', analysisParameters);
        }

        // Update UI elements with current parameter values
        updateUIFromParameters();

    } catch (error) {
        console.error('載入參數失敗:', error);
        console.log('使用預設參數');
    }
}

// Update UI elements from internal parameter object
function updateUIFromParameters() {
    console.log('🔄 更新UI顯示參數值...');

    Object.keys(analysisParameters).forEach(key => {
        const element = document.getElementById(key);
        if (element) {
            if (element.type === 'checkbox') {
                element.checked = analysisParameters[key];
            } else {
                element.value = analysisParameters[key];
            }
            console.log(`  - 設定 ${key} = ${analysisParameters[key]}`);
        }
    });
}

// Reset parameters to default values
function resetParametersToDefault() {
    console.log('🔄 重置參數到預設值...');

    // Reset to default values
    analysisParameters.enableSkyRemoval = true;
    analysisParameters.pointsPerSide = 48;
    analysisParameters.pointsPerBatch = 64;
    analysisParameters.predIouThresh = 0.75;
    analysisParameters.stabilityScoreThresh = 0.9;
    analysisParameters.stabilityScoreOffset = 1.0;
    analysisParameters.minMaskRegionArea = 10000;
    analysisParameters.iouThreshold = 0.3;
    analysisParameters.erosionKernelSize = 5;
    analysisParameters.dilationKernelSize = 5;
    analysisParameters.distanceThreshold = 20;
    analysisParameters.similarityThreshold = 0.35;

    console.log('✅ 預設參數:', analysisParameters);

    // Update UI elements
    updateUIFromParameters();

    // Save to localStorage
    localStorage.setItem('analysisParameters', JSON.stringify(analysisParameters));

    // Show confirmation
    showNotification('Parameters reset to default values!', 'info');
}

// Show notification message
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        animation: slideIn 0.3s ease;
        max-width: 300px;
    `;

    // Set background color based on type
    switch (type) {
        case 'success':
            notification.style.background = '#4CAF50';
            break;
        case 'error':
            notification.style.background = '#f44336';
            break;
        case 'warning':
            notification.style.background = '#ff9800';
            break;
        default:
            notification.style.background = '#2196F3';
    }

    notification.textContent = message;
    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// ===== Photo Selection Management =====
function showImageSelectionGrid() {
    const container = document.getElementById('photoSelectionContainer');
    if (!container) return;

    // If we have preview images, show selection grid
    if (previewImages && previewImages.length >= 2) {
        const gridHTML = `
            <div class="image-selection-grid">
                ${previewImages.map((img, index) => `
                    <div class="selectable-image ${selectedImagePair.includes(index) ? 'selected' : ''}"
                         onclick="selectImageForComparison(${index})">
                        <img src="${img}" alt="Image ${index + 1}">
                        ${selectedImagePair.includes(index) ?
                            `<div class="selection-number">${selectedImagePair.indexOf(index) + 1}</div>` :
                            ''}
                    </div>
                `).join('')}
            </div>
            <div style="margin-top: 15px; text-align: center; color: #666; font-size: 14px;">
                ${selectedImagePair.filter(x => x !== null).length}/2 photos selected for comparison
            </div>
        `;
        container.innerHTML = gridHTML;
    } else {
        // Show hint message
        container.innerHTML = `
            <div id="photoSelectionHint" style="text-align: center; color: #666; padding: 20px; font-style: italic;">
                📸 Upload multiple photos to see the selection grid here
            </div>
        `;
    }
}

// Handle image selection for comparison
function selectImageForComparison(imageIndex) {
    if (selectedImagePair[0] === imageIndex) {
        // Deselect if clicking the same image
        selectedImagePair[0] = selectedImagePair[1];
        selectedImagePair[1] = null;
    } else if (selectedImagePair[1] === imageIndex) {
        // Deselect second image
        selectedImagePair[1] = null;
    } else if (selectedImagePair[0] === null) {
        // Select as first image
        selectedImagePair[0] = imageIndex;
    } else if (selectedImagePair[1] === null) {
        // Select as second image
        selectedImagePair[1] = imageIndex;
    } else {
        // Replace first image and shift
        selectedImagePair[0] = selectedImagePair[1];
        selectedImagePair[1] = imageIndex;
    }

    // Update preview displays
    updatePreviewDisplays();

    // Refresh selection grid
    showImageSelectionGrid();

    // Show/hide advanced button based on selection
    const advancedBtn = document.getElementById('advancedBtn');
    if (advancedBtn) {
        advancedBtn.style.display = selectedImagePair.filter(x => x !== null).length === 2 ? 'inline-block' : 'none';
    }
}

// Enhanced clearAll function
function clearAll() {
    // 清除照片相關數據
    previewImages = [];
    selectedPhotos = [];
    selectedImagePair = [null, null];
    currentImageIndex = 0;

    // 重置文件輸入
    const photoInput = document.getElementById('photoInput');
    const photoFolder = document.getElementById('photoFolder');
    if (photoInput) photoInput.value = '';
    if (photoFolder) photoFolder.value = '';

    // 清除狀態顯示
    const photoStatus = document.getElementById('photoStatus');
    if (photoStatus) photoStatus.textContent = 'No photos selected yet';

    // 清除預覽區域
    const preview1 = document.getElementById('preview1');
    const preview2 = document.getElementById('preview2');
    if (preview1) {
        preview1.innerHTML = 'Image 1<br>No image selected yet';
        preview1.className = 'no-preview';
    }
    if (preview2) {
        preview2.innerHTML = 'Image 2<br>No image selected yet';
        preview2.className = 'no-preview';
    }

    // 隱藏導航控制
    const navigationControls = document.getElementById('navigationControls');
    if (navigationControls) navigationControls.style.display = 'none';

    // 清除照片選擇網格
    const photoSelectionContainer = document.getElementById('photoSelectionContainer');
    if (photoSelectionContainer) {
        photoSelectionContainer.innerHTML = `
            <div id="photoSelectionHint" style="text-align: center; color: #666; padding: 20px; font-style: italic;">
                📸 Upload multiple photos to see the selection grid here
            </div>
        `;
    }

    // 清除照片選擇網格（舊版）
    const existingGrid = document.getElementById('imageSelectionGrid');
    if (existingGrid) existingGrid.remove();

    // 重新顯示提示文字
    const hint = document.getElementById('photoSelectionHint');
    if (hint) hint.style.display = 'block';

    // 隱藏進度和結果區域
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    if (progressSection) progressSection.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'none';

    // 重置模式為進階模式
    setMode('advanced');

    console.log('✅ All data cleared');
    showAlert('All selections and results have been cleared', 'success');
}
