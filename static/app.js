/**
 * Manuscript Layout Detection — Web UI Application Logic
 * Handles file upload, drag-and-drop, API calls, result rendering,
 * tab switching, class distribution bars, and region table.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── DOM Elements ──
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const uploadSection = document.getElementById('upload-section');
    const processingSection = document.getElementById('processing-section');
    const resultsSection = document.getElementById('results-section');
    const progressFill = document.getElementById('progress-fill');
    const processingStatus = document.getElementById('processing-status');

    // Settings
    const confidenceSlider = document.getElementById('confidence-slider');
    const nmsSlider = document.getElementById('nms-slider');
    const confidenceValue = document.getElementById('confidence-value');
    const nmsValue = document.getElementById('nms-value');

    // Stats
    const statRegions = document.getElementById('stat-regions');
    const statTime = document.getElementById('stat-time');
    const statSize = document.getElementById('stat-size');
    const statFilename = document.getElementById('stat-filename');

    // Images & JSON
    const annotatedImg = document.getElementById('annotated-img');
    const originalImg = document.getElementById('original-img');
    const jsonOutput = document.getElementById('json-output');

    // Class distribution
    const classBars = document.getElementById('class-bars');

    // Region table
    const regionsTbody = document.getElementById('regions-tbody');
    const tableFilters = document.getElementById('table-filters');

    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');

    // Pipeline steps
    const pipelineSteps = document.querySelectorAll('.pipeline-step');

    // New analysis
    const newAnalysisBtn = document.getElementById('new-analysis-btn');

    // Class color map
    const CLASS_COLORS = {
        header: '#ff6b6b',
        footer: '#00b4d8',
        main_text: '#2ecc71',
        side_text: '#9b59b6',
        filler: '#f39c12'
    };

    let allRegions = [];

    // ── Settings Sliders ──
    confidenceSlider.addEventListener('input', () => {
        confidenceValue.textContent = parseFloat(confidenceSlider.value).toFixed(2);
    });
    nmsSlider.addEventListener('input', () => {
        nmsValue.textContent = parseFloat(nmsSlider.value).toFixed(2);
    });

    // ── File Upload ──
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // ── Drag & Drop ──
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // ── Tab Switching ──
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.image-panel').forEach(p => p.classList.remove('active'));
            document.getElementById(`panel-${tab}`).classList.add('active');
        });
    });

    // ── New Analysis ──
    newAnalysisBtn.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        uploadSection.style.display = '';
        fileInput.value = '';
        resetPipelineSteps();
    });

    // ── Pipeline Animation ──
    function setPipelineStep(stepNum, state) {
        pipelineSteps.forEach(step => {
            const num = parseInt(step.dataset.step);
            if (num < stepNum) {
                step.classList.remove('active');
                step.classList.add('done');
            } else if (num === stepNum) {
                step.classList.remove('done');
                step.classList.add(state);
            } else {
                step.classList.remove('active', 'done');
            }
        });
    }

    function resetPipelineSteps() {
        pipelineSteps.forEach(step => {
            step.classList.remove('active', 'done');
        });
    }

    function allPipelineDone() {
        pipelineSteps.forEach(step => {
            step.classList.remove('active');
            step.classList.add('done');
        });
    }

    // ── Handle File Upload & Processing ──
    async function handleFile(file) {
        const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
        if (!allowed.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|bmp|tif|tiff|webp)$/i)) {
            alert('Unsupported file format. Please upload a JPG, PNG, BMP, TIFF, or WebP image.');
            return;
        }

        // Show processing
        uploadSection.style.display = 'none';
        processingSection.style.display = '';
        resultsSection.style.display = 'none';

        // Animate pipeline
        let progress = 0;
        progressFill.style.width = '0%';

        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.random() * 3;
                progressFill.style.width = `${Math.min(progress, 90)}%`;
            }
        }, 300);

        const stages = [
            { step: 1, msg: 'Stage 1: Preprocessing (CLAHE, Deskew, Sauvola)...', delay: 500 },
            { step: 2, msg: 'Stage 2: Region Proposal (EasyOCR + MSER)...', delay: 2000 },
            { step: 3, msg: 'Stage 3: Classification (16D Features + GBM)...', delay: 4000 },
            { step: 4, msg: 'Stage 4: Postprocessing (NMS, Clipping)...', delay: 6000 }
        ];

        stages.forEach(({ step, msg, delay }) => {
            setTimeout(() => {
                setPipelineStep(step, 'active');
                processingStatus.textContent = msg;
            }, delay);
        });

        // Build FormData
        const formData = new FormData();
        formData.append('image', file);
        formData.append('confidence_threshold', confidenceSlider.value);
        formData.append('nms_threshold', nmsSlider.value);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            progressFill.style.width = '100%';
            allPipelineDone();

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Processing failed');
            }

            const data = await response.json();

            // Short pause for visual effect
            await new Promise(resolve => setTimeout(resolve, 600));

            // Switch to results
            processingSection.style.display = 'none';
            resultsSection.style.display = '';

            renderResults(data);

        } catch (error) {
            clearInterval(progressInterval);
            processingSection.style.display = 'none';
            uploadSection.style.display = '';
            resetPipelineSteps();
            alert(`Error: ${error.message}`);
        }
    }

    // ── Render Results ──
    function renderResults(data) {
        // Stats
        animateNumber(statRegions, data.total_regions);
        statTime.textContent = `${data.processing_time}s`;
        statSize.textContent = `${data.image_size.width}×${data.image_size.height}`;
        statFilename.textContent = data.image_name;

        // Images
        annotatedImg.src = `data:image/jpeg;base64,${data.annotated_image}`;
        originalImg.src = `data:image/jpeg;base64,${data.original_image}`;

        // JSON
        const jsonData = {
            image_name: data.image_name,
            image_size: data.image_size,
            processing_time_seconds: data.processing_time,
            regions_count: data.total_regions,
            regions: data.regions
        };
        jsonOutput.textContent = JSON.stringify(jsonData, null, 2);

        // Class Distribution
        renderClassDistribution(data.class_counts, data.total_regions);

        // Store all regions
        allRegions = data.regions;

        // Region Table & Filters
        renderFilters(data.class_counts);
        renderRegionTable(data.regions);

        // Reset tabs to annotated
        tabBtns.forEach(b => b.classList.remove('active'));
        document.getElementById('tab-annotated').classList.add('active');
        document.querySelectorAll('.image-panel').forEach(p => p.classList.remove('active'));
        document.getElementById('panel-annotated').classList.add('active');
    }

    // ── Animate Counter ──
    function animateNumber(el, target) {
        let current = 0;
        const step = Math.max(1, Math.ceil(target / 40));
        const interval = setInterval(() => {
            current += step;
            if (current >= target) {
                current = target;
                clearInterval(interval);
            }
            el.textContent = current;
        }, 25);
    }

    // ── Class Distribution Bars ──
    function renderClassDistribution(classCounts, total) {
        classBars.innerHTML = '';

        const allClasses = ['header', 'footer', 'main_text', 'side_text', 'filler'];

        allClasses.forEach(cls => {
            const count = classCounts[cls] || 0;
            const pct = total > 0 ? (count / total) * 100 : 0;

            const row = document.createElement('div');
            row.className = 'class-bar-row';

            row.innerHTML = `
                <div class="class-label">
                    <span class="class-dot" style="background: ${CLASS_COLORS[cls]}"></span>
                    ${cls.replace('_', ' ')}
                </div>
                <div class="class-bar-track">
                    <div class="class-bar-fill" style="background: ${CLASS_COLORS[cls]}; width: 0%"></div>
                </div>
                <span class="class-count">${count}</span>
            `;

            classBars.appendChild(row);

            // Animate bar width after append
            requestAnimationFrame(() => {
                setTimeout(() => {
                    row.querySelector('.class-bar-fill').style.width = `${pct}%`;
                }, 100);
            });
        });
    }

    // ── Filter Buttons ──
    function renderFilters(classCounts) {
        tableFilters.innerHTML = '';

        const allBtn = document.createElement('button');
        allBtn.className = 'filter-btn active';
        allBtn.textContent = 'All';
        allBtn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            allBtn.classList.add('active');
            renderRegionTable(allRegions);
        });
        tableFilters.appendChild(allBtn);

        Object.keys(classCounts).sort().forEach(cls => {
            const btn = document.createElement('button');
            btn.className = 'filter-btn';
            btn.textContent = `${cls.replace('_', ' ')} (${classCounts[cls]})`;
            btn.addEventListener('click', () => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                renderRegionTable(allRegions.filter(r => r.label === cls));
            });
            tableFilters.appendChild(btn);
        });
    }

    // ── Region Table ──
    function renderRegionTable(regions) {
        regionsTbody.innerHTML = '';

        regions.forEach((region, idx) => {
            const [x1, y1, x2, y2] = region.box;
            const area = (x2 - x1) * (y2 - y1);
            const confPct = (region.score * 100).toFixed(1);

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="color: var(--text-muted); font-family: var(--font-mono); font-size: 0.78rem;">${idx + 1}</td>
                <td><span class="label-tag ${region.label}"><span class="class-dot" style="background: ${CLASS_COLORS[region.label]}; width: 7px; height: 7px;"></span>${region.label.replace('_', ' ')}</span></td>
                <td class="box-coords">[${x1}, ${y1}, ${x2}, ${y2}]</td>
                <td>
                    <div class="confidence-bar">
                        <div class="conf-track"><div class="conf-fill" style="width: ${confPct}%"></div></div>
                        <span class="conf-value">${confPct}%</span>
                    </div>
                </td>
                <td style="font-family: var(--font-mono); font-size: 0.82rem; color: var(--text-secondary);">${area.toLocaleString()}</td>
            `;
            regionsTbody.appendChild(tr);
        });
    }
});
