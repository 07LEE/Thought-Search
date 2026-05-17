let globalNodes = [];
let globalEdges = [];
let originalEdgeX = [], originalEdgeY = [], originalEdgeZ = [];
let isUpdating = false;
let currentThreshold = 0.80;
let lastHighlightedIndex = null;
let isAutoOrbit = true;
let orbitAngle = 0;
let activeSubHighlight = null;
let timelineStart = null;
let timelineEnd = null;
let minTimelineTime = null;
let maxTimelineTime = null;
let activeSearchIndices = null;

function getNodeTime(node) {
    if (node.metadata && node.metadata.date) {
        const d = new Date(node.metadata.date);
        if (!isNaN(d.getTime())) return d.getTime();
    }
    if (node.mtime) {
        return node.mtime * 1000;
    }
    return new Date().getTime();
}

function formatDate(timestamp) {
    const d = new Date(timestamp);
    if (isNaN(d.getTime())) return "YYYY-MM-DD";
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd}`;
}

function applyGraphVisualState() {
    const startVal = timelineStart !== null ? timelineStart : minTimelineTime;
    const endVal = timelineEnd !== null ? timelineEnd : maxTimelineTime;
    
    let baseOpacities = [];
    let baseSizes = [];
    
    if (lastHighlightedIndex !== null) {
        const connectedIndices = new Set();
        globalEdges.forEach(edge => {
            const [sIdx, tIdx, score] = edge;
            if (score >= currentThreshold && (sIdx === lastHighlightedIndex || tIdx === lastHighlightedIndex)) {
                const neighborIdx = (sIdx === lastHighlightedIndex ? tIdx : sIdx);
                connectedIndices.add(neighborIdx);
            }
        });
        
        baseOpacities = globalNodes.map((n, i) => (i === lastHighlightedIndex || connectedIndices.has(i)) ? 1.0 : 0.05);
        baseSizes = globalNodes.map((n, i) => (i === lastHighlightedIndex || connectedIndices.has(i)) ? n.size * 1.5 : 2);
    } else if (activeSubHighlight !== null) {
        const [cat, sub] = activeSubHighlight.split(' > ');
        baseOpacities = globalNodes.map(n => {
            const cs = n.metadata.categories || ["Uncategorized"];
            return (cs[0] === cat && (cs[1] || "General") === sub) ? 1.0 : 0.08;
        });
        baseSizes = globalNodes.map(n => {
            const cs = n.metadata.categories || ["Uncategorized"];
            const isMatch = cs[0] === cat && (cs[1] || "General") === sub;
            return isMatch ? n.size : 1.5;
        });
    } else if (activeSearchIndices !== null) {
        baseOpacities = globalNodes.map((n, i) => activeSearchIndices.has(i) ? 1.0 : 0.05);
        baseSizes = globalNodes.map((n, i) => activeSearchIndices.has(i) ? n.size * 1.5 : 2);
    } else {
        baseOpacities = globalNodes.map(() => 0.9);
        baseSizes = globalNodes.map(n => n.size);
    }
    
    const finalOpacities = globalNodes.map((n, i) => {
        const time = getNodeTime(n);
        if (time >= startVal && time <= endVal) {
            return baseOpacities[i];
        } else {
            return 0.005;
        }
    });
    
    const finalSizes = globalNodes.map((n, i) => {
        const time = getNodeTime(n);
        if (time >= startVal && time <= endVal) {
            return baseSizes[i];
        } else {
            return 0.5;
        }
    });
    
    // Compute edges based on threshold and timeline
    const edgeX = [], edgeY = [], edgeZ = [];
    globalEdges.forEach(edge => {
        const [sourceIdx, targetIdx, score] = edge;
        if (score >= currentThreshold) {
            const s = globalNodes[sourceIdx], t = globalNodes[targetIdx];
            const sTime = getNodeTime(s);
            const tTime = getNodeTime(t);
            if (sTime >= startVal && sTime <= endVal && tTime >= startVal && tTime <= endVal) {
                edgeX.push(s.x, t.x, null); edgeY.push(s.y, t.y, null); edgeZ.push(s.z, t.z, null);
            }
        }
    });
    
    let finalEdgeX = edgeX;
    let finalEdgeY = edgeY;
    let finalEdgeZ = edgeZ;
    let edgeColor = 'rgba(56, 189, 248, 0.15)';
    let edgeWidth = 1;
    
    if (lastHighlightedIndex !== null) {
        const hX = [], hY = [], hZ = [];
        globalEdges.forEach(edge => {
            const [sIdx, tIdx, score] = edge;
            if (score >= currentThreshold && (sIdx === lastHighlightedIndex || tIdx === lastHighlightedIndex)) {
                const s = globalNodes[sIdx], t = globalNodes[tIdx];
                const sTime = getNodeTime(s);
                const tTime = getNodeTime(t);
                if (sTime >= startVal && sTime <= endVal && tTime >= startVal && tTime <= endVal) {
                    hX.push(s.x, t.x, null); hY.push(s.y, t.y, null); hZ.push(s.z, t.z, null);
                }
            }
        });
        finalEdgeX = hX.length ? hX : [null];
        finalEdgeY = hY.length ? hY : [null];
        finalEdgeZ = hZ.length ? hZ : [null];
        edgeColor = 'rgba(56, 189, 248, 0.8)';
        edgeWidth = 3;
    }
    
    Plotly.restyle('plot', {
        'marker.opacity': [finalOpacities],
        'marker.size': [finalSizes]
    }, [1]);
    
    Plotly.restyle('plot', {
        'x': [finalEdgeX],
        'y': [finalEdgeY],
        'z': [finalEdgeZ],
        'line.color': edgeColor,
        'line.width': edgeWidth
    }, [0]);
}

// --- Color Space Utility Helpers ---
function hexToHls(hex) {
    hex = hex.replace(/^#/, '');
    let r = parseInt(hex.substring(0, 2), 16) / 255;
    let g = parseInt(hex.substring(2, 4), 16) / 255;
    let b = parseInt(hex.substring(4, 6), 16) / 255;
    
    let max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, l, s;
    l = (max + min) / 2;
    
    if (max === min) {
        h = s = 0;
    } else {
        let d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    return { h, l, s };
}

function hlsToHex(h, l, s) {
    let r, g, b;
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        let q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        let p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }
    const toHex = x => {
        const hex = Math.round(x * 255).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    };
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function updateNodeColorsForCategory(targetCat, newHex) {
    const targetNodes = globalNodes.filter(n => n.category === targetCat);
    const subs = Array.from(new Set(targetNodes.map(n => {
        const cats = n.metadata.categories || ["Uncategorized"];
        return cats[1] || "General";
    }))).sort();
    
    const { h, l, s } = hexToHls(newHex);
    const numSubs = subs.length;
    
    const subToColor = {};
    subs.forEach((sub, subIdx) => {
        if (numSubs <= 1) {
            subToColor[sub] = newHex;
        } else {
            const lightnessOffset = -0.22 + (subIdx * 0.44 / (numSubs - 1));
            const saturationOffset = -0.15 + (subIdx * 0.25 / (numSubs - 1));
            const modulatedL = Math.max(0.30, Math.min(0.90, l + lightnessOffset));
            const modulatedS = Math.max(0.35, Math.min(1.0, s + saturationOffset));
            subToColor[sub] = hlsToHex(h, modulatedL, modulatedS);
        }
    });
    
    globalNodes.forEach(node => {
        if (node.category === targetCat) {
            const cats = node.metadata.categories || ["Uncategorized"];
            const sub = cats[1] || "General";
            node.color = subToColor[sub] || newHex;
        }
    });
}

async function init() {
    try {
        // --- Markdown & Table Setup ---
        marked.use({
            hooks: {
                postprocess(html) {
                    return html.replace(/<table>/g, '<div class="table-wrapper"><table>')
                               .replace(/<\/table>/g, '</table></div>');
                }
            }
        });

        marked.setOptions({ 
            gfm: true, 
            breaks: true, 
            headerIds: false, 
            mangle: false 
        });

        const response = await fetch('/data/viz-data.json');
        const data = await response.json();
        globalNodes = data.nodes;
        globalEdges = data.edges;
        const categories = data.categories;

        // 로컬 스토리지 결합 및 노드 초기 색상 보정
        const savedColorsRaw = localStorage.getItem('thought_search_custom_colors');
        const customColors = savedColorsRaw ? JSON.parse(savedColorsRaw) : {};
        
        for (const [cat, customHex] of Object.entries(customColors)) {
            if (categories[cat]) {
                categories[cat] = customHex;
                updateNodeColorsForCategory(cat, customHex);
            }
        }

        const legendContainer = document.getElementById('legend');
        legendContainer.innerHTML = '';
        
        const legendHeaderWrapper = document.createElement('div');
        legendHeaderWrapper.style.display = 'flex';
        legendHeaderWrapper.style.justifyContent = 'space-between';
        legendHeaderWrapper.style.alignItems = 'center';
        legendHeaderWrapper.style.marginBottom = '8px';
        
        const legendTitleEl = document.createElement('span');
        legendTitleEl.style.fontSize = '0.7rem';
        legendTitleEl.style.color = '#64748b';
        legendTitleEl.style.textTransform = 'uppercase';
        legendTitleEl.style.letterSpacing = '0.05em';
        legendTitleEl.textContent = 'Categories';
        
        const resetColorsBtn = document.createElement('span');
        resetColorsBtn.className = 'legend-reset-btn';
        resetColorsBtn.textContent = 'Reset';
        resetColorsBtn.title = '색상 테마 초기화';
        
        resetColorsBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            localStorage.removeItem('thought_search_custom_colors');
            location.reload();
        });
        
        legendHeaderWrapper.appendChild(legendTitleEl);
        legendHeaderWrapper.appendChild(resetColorsBtn);
        legendContainer.appendChild(legendHeaderWrapper);

        // 1. 대분류별 소분류 그룹 정리
        const parentToSubs = {};
        globalNodes.forEach(n => {
            const cats = n.metadata.categories || ["Uncategorized"];
            const parent = cats[0];
            const sub = cats[1] || "General";
            if (!parentToSubs[parent]) {
                parentToSubs[parent] = new Set();
            }
            parentToSubs[parent].add(sub);
        });

        // 2. 아코디언 범례 렌더링
        for (const [cat, color] of Object.entries(categories)) {
            const group = document.createElement('div');
            group.className = 'legend-group';

            const header = document.createElement('div');
            header.className = 'legend-header';

            const colorBox = document.createElement('div');
            colorBox.className = 'legend-color';
            colorBox.style.background = color;
            colorBox.title = '대분류 색상 변경';
            
            // 색상 피커 기능 연동 및 이벤트 전파 차단
            colorBox.addEventListener('click', function(e) {
                e.stopPropagation();
                
                const picker = document.createElement('input');
                picker.type = 'color';
                picker.value = categories[cat];
                
                picker.addEventListener('input', function() {
                    const newColor = picker.value;
                    categories[cat] = newColor;
                    colorBox.style.background = newColor;
                    
                    customColors[cat] = newColor;
                    localStorage.setItem('thought_search_custom_colors', JSON.stringify(customColors));
                    
                    updateNodeColorsForCategory(cat, newColor);
                    
                    Plotly.restyle('plot', {
                        'marker.color': [globalNodes.map(n => n.color)]
                    }, [1]);

                    // 하위 소분류 미니 색상 칩들도 즉시 리페인팅
                    const subList = group.querySelector('.legend-sub-list');
                    if (subList) {
                        const subItems = subList.querySelectorAll('.legend-sub-item');
                        subItems.forEach(item => {
                            const subName = item.querySelector('span').textContent;
                            const subColorBox = item.querySelector('.legend-sub-color');
                            if (subColorBox) {
                                subColorBox.style.background = getSubColor(cat, subName);
                            }
                        });
                    }
                });
                
                picker.click();
            });

            // 소분류 그라데이션 색상 동적 획득 헬퍼
            const getSubColor = (parent, sub) => {
                const foundNode = globalNodes.find(n => {
                    const cs = n.metadata.categories || ["Uncategorized"];
                    return cs[0] === parent && (cs[1] || "General") === sub;
                });
                return foundNode ? foundNode.color : categories[parent];
            };

            const label = document.createElement('span');
            label.className = 'legend-title';
            label.textContent = cat;

            const arrow = document.createElement('span');
            arrow.className = 'legend-arrow';
            arrow.textContent = '▼';

            header.appendChild(colorBox);
            header.appendChild(label);
            header.appendChild(arrow);

            // 하위 소분류 목록 리스트
            const subList = document.createElement('div');
            subList.className = 'legend-sub-list';

            const subsArray = Array.from(parentToSubs[cat] || []).sort();
            subsArray.forEach(sub => {
                const subItem = document.createElement('div');
                subItem.className = 'legend-sub-item';

                const subColorBox = document.createElement('div');
                subColorBox.className = 'legend-sub-color';
                subColorBox.style.background = getSubColor(cat, sub);

                const subLabel = document.createElement('span');
                subLabel.textContent = sub;

                subItem.appendChild(subColorBox);
                subItem.appendChild(subLabel);

                // 소분류 지식 노드 실시간 하이라이팅 필터 장착
                subItem.addEventListener('click', function(e) {
                    e.stopPropagation();
                    
                    const subKey = `${cat} > ${sub}`;
                    const allSubItems = legendContainer.querySelectorAll('.legend-sub-item');
                    
                    if (activeSubHighlight === subKey) {
                        // 이미 선택된 항목 재클릭 시 필터링 전면 해제
                        activeSubHighlight = null;
                        subItem.classList.remove('active');
                        applyGraphVisualState();
                    } else {
                        // 새로운 소분류 공간 하이라이트 가동
                        activeSubHighlight = subKey;
                        allSubItems.forEach(item => item.classList.remove('active'));
                        subItem.classList.add('active');
                        applyGraphVisualState();
                    }
                });

                subList.appendChild(subItem);
            });

            // 헤더 영역 클릭 시 아코디언 토글 적용
            header.addEventListener('click', function() {
                group.classList.toggle('active');
            });

            group.appendChild(header);
            if (subsArray.length > 0) {
                group.appendChild(subList);
            }
            legendContainer.appendChild(group);
        }

        const nodeTrace = {
            x: globalNodes.map(n => n.x),
            y: globalNodes.map(n => n.y),
            z: globalNodes.map(n => n.z),
            text: globalNodes.map(n => n.metadata.title || n.metadata.filename),
            mode: 'markers',
            type: 'scatter3d',
            marker: {
                size: globalNodes.map(n => n.size),
                color: globalNodes.map(n => n.color),
                opacity: 0.9,
                line: { color: 'rgba(255, 255, 255, 0.1)', width: 0.5 }
            },
            hoverinfo: 'text'
        };


        const getFilteredEdges = (threshold, start, end) => {
            const edgeX = [], edgeY = [], edgeZ = [];
            const sVal = start !== undefined ? start : minTimelineTime;
            const eVal = end !== undefined ? end : maxTimelineTime;
            globalEdges.forEach(edge => {
                const [sourceIdx, targetIdx, score] = edge;
                if (score >= threshold) {
                    const s = globalNodes[sourceIdx], t = globalNodes[targetIdx];
                    const sTime = getNodeTime(s);
                    const tTime = getNodeTime(t);
                    if (sTime >= sVal && sTime <= eVal && tTime >= sVal && tTime <= eVal) {
                        edgeX.push(s.x, t.x, null); edgeY.push(s.y, t.y, null); edgeZ.push(s.z, t.z, null);
                    }
                }
            });
            return { x: edgeX, y: edgeY, z: edgeZ };
        };

        // --- Timeline Initial Range Configuration ---
        const nodeTimes = globalNodes.map(getNodeTime);
        minTimelineTime = Math.min(...nodeTimes);
        maxTimelineTime = Math.max(...nodeTimes);
        
        if (minTimelineTime === maxTimelineTime) {
            minTimelineTime -= 24 * 60 * 60 * 1000;
            maxTimelineTime += 24 * 60 * 60 * 1000;
        }
        
        timelineStart = minTimelineTime;
        timelineEnd = maxTimelineTime;
        
        const startSlider = document.getElementById('timeline-start-slider');
        const endSlider = document.getElementById('timeline-end-slider');
        
        startSlider.min = minTimelineTime;
        startSlider.max = maxTimelineTime;
        startSlider.value = minTimelineTime;
        startSlider.step = 24 * 60 * 60 * 1000;
        
        endSlider.min = minTimelineTime;
        endSlider.max = maxTimelineTime;
        endSlider.value = maxTimelineTime;
        endSlider.step = 24 * 60 * 60 * 1000;
        
        document.getElementById('timeline-start-val').textContent = formatDate(minTimelineTime);
        document.getElementById('timeline-end-val').textContent = formatDate(maxTimelineTime);
        
        startSlider.addEventListener('input', function() {
            let val = parseFloat(this.value);
            if (val > parseFloat(endSlider.value)) {
                this.value = endSlider.value;
                val = parseFloat(endSlider.value);
            }
            timelineStart = val;
            document.getElementById('timeline-start-val').textContent = formatDate(val);
            applyGraphVisualState();
        });
        
        endSlider.addEventListener('input', function() {
            let val = parseFloat(this.value);
            if (val < parseFloat(startSlider.value)) {
                this.value = startSlider.value;
                val = parseFloat(startSlider.value);
            }
            timelineEnd = val;
            document.getElementById('timeline-end-val').textContent = formatDate(val);
            applyGraphVisualState();
        });

        const initialEdges = getFilteredEdges(currentThreshold, minTimelineTime, maxTimelineTime);
        originalEdgeX = initialEdges.x;
        originalEdgeY = initialEdges.y;
        originalEdgeZ = initialEdges.z;

        const edgeTrace = {
            x: originalEdgeX, y: originalEdgeY, z: originalEdgeZ,
            mode: 'lines', type: 'scatter3d',
            line: { color: 'rgba(56, 189, 248, 0.15)', width: 1 },
            hoverinfo: 'none'
        };

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 0, r: 0, b: 0, l: 0 },
            scene: {
                xaxis: { showgrid: false, zeroline: false, showticklabels: false, title: '' },
                yaxis: { showgrid: false, zeroline: false, showticklabels: false, title: '' },
                zaxis: { showgrid: false, zeroline: false, showticklabels: false, title: '' },
                camera: { eye: { x: 1.8, y: 1.8, z: 1.8 } }
            },
            showlegend: false
        };

        Plotly.newPlot('plot', [edgeTrace, nodeTrace], layout, { responsive: true, displayModeBar: false });

        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');
        let searchTimeout;

        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            const query = this.value.trim();
            
            if (!query) {
                searchResults.classList.remove('active');
                searchResults.innerHTML = '';
                resetView();
                return;
            }

            searchTimeout = setTimeout(async () => {
                try {
                    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&k=10`);
                    const data = await response.json();
                    
                    if (data.status === 'success' && data.results.length > 0) {
                        displaySearchResults(data.results);
                    } else {
                        searchResults.innerHTML = '<div style="padding: 15px; color: #64748b; font-size: 0.8rem; text-align: center;">검색 결과가 없습니다.</div>';
                        searchResults.classList.add('active');
                    }
                } catch (err) {
                    console.error('Search error:', err);
                }
            }, 300);
        });

        function displaySearchResults(results) {
            searchResults.innerHTML = '';
            searchResults.classList.add('active');
            
            const resultPaths = new Set(results.map(res => res.metadata.rel_path));
            const resultIndices = new Set();
            
            globalNodes.forEach((node, i) => {
                if (resultPaths.has(node.metadata.rel_path)) {
                    resultIndices.add(i);
                }
            });

            results.forEach(res => {
                const item = document.createElement('div');
                item.className = 'search-result-item';
                
                const meta = res.metadata;
                const score = res.rerank_score || res.score;
                const title = meta.title || meta.filename;
                
                item.innerHTML = `
                    <div class="result-title">${title}</div>
                    <div class="result-snippet">${res.text}</div>
                    <div class="result-meta">
                        <div class="result-score">Similarity: ${score.toFixed(4)}</div>
                    </div>
                `;
                
                item.onclick = () => {
                    const nodeIndex = globalNodes.findIndex(n => n.metadata.rel_path === meta.rel_path);
                    if (nodeIndex !== -1) {
                        highlightNode(nodeIndex);
                        searchResults.classList.remove('active');
                    } else {
                        alert('Graph에서 해당 문서를 찾을 수 없습니다.');
                    }
                };
                
                searchResults.appendChild(item);
            });

            // Update graph to highlight search results
            if (resultIndices.size > 0) {
                activeSearchIndices = resultIndices;
                applyGraphVisualState();
            }
        }

        // Close search results when clicking outside
        document.addEventListener('click', (e) => {
            if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
                searchResults.classList.remove('active');
            }
        });

        const thresholdSlider = document.getElementById('threshold-slider');
        const thresholdVal = document.getElementById('threshold-val');
        
        thresholdSlider.addEventListener('input', function() {
            currentThreshold = parseFloat(this.value);
            thresholdVal.textContent = currentThreshold.toFixed(2);
            applyGraphVisualState();
        });

        const plotDiv = document.getElementById('plot');
        
        // --- Auto-Orbit Logic ---
        const orbitToggle = document.getElementById('orbit-toggle');
        let isUserInteracting = false;

        orbitToggle.addEventListener('change', function() {
            isAutoOrbit = this.checked;
            if (isAutoOrbit) requestAnimationFrame(animate);
        });

        plotDiv.addEventListener('mousedown', () => { isUserInteracting = true; });
        window.addEventListener('mouseup', () => { isUserInteracting = false; });

        // Handle Mouse Wheel (Zoom)
        let wheelActive = false;
        let wheelTimer;
        plotDiv.addEventListener('wheel', () => {
            wheelActive = true;
            clearTimeout(wheelTimer);
            wheelTimer = setTimeout(() => { wheelActive = false; }, 150);
        }, { passive: true });

        // Sync camera state on manual change
        plotDiv.on('plotly_relayout', function(eventData) {
            const eye = eventData['scene.camera.eye'] || (eventData['scene.camera'] && eventData['scene.camera'].eye);
            if (eye) {
                orbitAngle = Math.atan2(eye.y, eye.x);
            }
        });

        let frameCount = 0;
        async function animate() {
            if (!isAutoOrbit || lastHighlightedIndex !== null || isUserInteracting) {
                if (isAutoOrbit) requestAnimationFrame(animate);
                return;
            }
            
            // If wheeling, slow down update rate significantly to prevent jitter
            frameCount++;
            const throttle = wheelActive ? 5 : 2; 
            if (frameCount % throttle !== 0) {
                requestAnimationFrame(animate);
                return;
            }

            const currentLayout = plotDiv.layout;
            if (!currentLayout?.scene?.camera?.eye) {
                requestAnimationFrame(animate);
                return;
            }

            const currentEye = currentLayout.scene.camera.eye;
            const radius = Math.sqrt(currentEye.x ** 2 + currentEye.y ** 2);
            
            // Increment angle
            orbitAngle += wheelActive ? 0.001 : 0.004;
            
            const x = radius * Math.cos(orbitAngle);
            const y = radius * Math.sin(orbitAngle);
            
            try {
                // Use await to ensure the browser finishes rendering before next frame
                await Plotly.relayout('plot', {
                    'scene.camera.eye': { x: x, y: y, z: currentEye.z }
                });
            } catch (err) {}
            
            if (isAutoOrbit) requestAnimationFrame(animate);
        }

        // Start animation after a short delay
        setTimeout(() => { if (isAutoOrbit) requestAnimationFrame(animate); }, 1000);

        plotDiv.on('plotly_click', function(eventData) {
            if (isUpdating) return;
            if (!eventData || !eventData.points || eventData.points.length === 0) { resetView(); return; }
            const p = eventData.points[0];
            if (p.fullData.mode !== 'markers') { resetView(); return; }
            highlightNode(p.pointNumber);
        });

        document.getElementById('close-panel-btn').onclick = resetView;
        window.addEventListener('keydown', e => { if (e.key === 'Escape') resetView(); });

        document.getElementById('doc-count').textContent = `Stored: ${globalNodes.length} thoughts`;

        const syncBtn = document.getElementById('sync-btn');
        syncBtn.addEventListener('click', async () => {
            syncBtn.disabled = true;
            syncBtn.classList.add('loading');
            syncBtn.innerHTML = 'Syncing...';
            
            try {
                const res = await fetch('/api/sync', { method: 'POST' });
                const result = await res.json();
                if (result.status === 'success') {
                    // Success! Reload the data.
                    // We can just re-init everything or reload the page.
                    // Reloading the page is simpler for now to ensure everything is fresh.
                    location.reload();
                } else {
                    alert('Sync failed: ' + result.message);
                }
            } catch (err) {
                console.error('Sync Error:', err);
                alert('An error occurred during sync.');
            } finally {
                syncBtn.disabled = false;
                syncBtn.classList.remove('loading');
                syncBtn.innerHTML = 'Sync DB';
            }
        });

    } catch (error) {
        console.error('Error:', error);
    }
}

async function highlightNode(index) {
    if (isUpdating && lastHighlightedIndex !== index) return;
    isUpdating = true;
    lastHighlightedIndex = index;
    try {
        const item = globalNodes[index];
        if (!item) return;
        
        let mathBlocks = [];

        document.getElementById('info-title').textContent = item.metadata.title || item.metadata.filename;
        document.getElementById('info-path').textContent = item.metadata.rel_path;
        
        // --- Image Path Resolution ---
        let processedContent = item.text;
        const docPath = item.metadata.rel_path;
        const docDir = docPath.substring(0, docPath.lastIndexOf('/') + 1);
        
        processedContent = processedContent.replace(/!\[(.*?)\]\((.*?)\)/g, (match, alt, src) => {
            if (src.startsWith('http') || src.startsWith('/')) return match;
            let fullSrc = "/posts/" + docDir + src;
            const parts = fullSrc.split('/');
            const stack = [];
            for (const part of parts) {
                if (part === '..') stack.pop();
                else if (part !== '.') stack.push(part);
            }
            return `![${alt}](${stack.join('/')})`;
        });

        // Remove context headings added by indexer (e.g., [# Heading])
        processedContent = processedContent.replace(/^\[#{1,6}.+?\]\n+/gm, '');
        // Protect display math ($$, \[...\])
        processedContent = processedContent.replace(/\$\$(.*?)\$\$|\\\[(.*?)\\\]/gs, (match) => {
            const id = `@@MATH_DISPLAY${mathBlocks.length}@@`;
            mathBlocks.push(match);
            return id;
        });

        // Protect inline math ($, \(...\)) - Do not use 's' flag for single $ to avoid eating multi-line blocks
        processedContent = processedContent.replace(/\$([^$\n]+?)\$|\\\((.*?)\\\)/g, (match) => {
            const id = `@@MATH_INLINE${mathBlocks.length}@@`;
            mathBlocks.push(match);
            return id;
        });

        const infoContent = document.getElementById('info-content');
        let html = marked.parse(processedContent.substring(0, 5000));
        
        // --- Math Restoration: Restore math tokens ---
        html = html.replace(/@@MATH_(DISPLAY|INLINE)(\d+)@@/g, (match, type, id) => {
            return mathBlocks[parseInt(id)];
        });
        
        infoContent.innerHTML = html;
        
        // --- Render Math using KaTeX auto-render ---
        renderMathInElement(infoContent, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\(', right: '\\)', display: false},
                {left: '\\[', right: '\\]', display: true}
            ],
            throwOnError : false
        });
        
        const tagsContainer = document.getElementById('info-tags');
        tagsContainer.innerHTML = '';
        (item.metadata.tags || []).forEach(tag => {
            const span = document.createElement('span');
            span.className = 'tag';
            span.textContent = tag;
            tagsContainer.appendChild(span);
        });

        const connectedIndices = new Set();
        const hX = [], hY = [], hZ = [];
        globalEdges.forEach(edge => {
            const [sIdx, tIdx, score] = edge;
            if (score >= currentThreshold && (sIdx === index || tIdx === index)) {
                const neighborIdx = (sIdx === index ? tIdx : sIdx);
                connectedIndices.add(neighborIdx);
                const s = globalNodes[sIdx], t = globalNodes[tIdx];
                hX.push(s.x, t.x, null); hY.push(s.y, t.y, null); hZ.push(s.z, t.z, null);
            }
        });

        const relatedList = document.getElementById('related-list');
        relatedList.innerHTML = '';
        if (connectedIndices.size === 0) {
            relatedList.innerHTML = '<div style="font-size: 0.75rem; color: #64748b;">연결된 지식이 없습니다.</div>';
        } else {
            connectedIndices.forEach(idx => {
                const neighbor = globalNodes[idx];
                const div = document.createElement('div');
                div.className = 'related-item';
                div.textContent = neighbor.metadata.title || neighbor.metadata.filename;
                div.onclick = (e) => { e.stopPropagation(); highlightNode(idx); };
                relatedList.appendChild(div);
            });
        }

        applyGraphVisualState();

        document.getElementById('info-panel').classList.add('active');
    } catch (e) { console.error('Highlight error:', e); }
    finally { setTimeout(() => { isUpdating = false; }, 100); }
}

async function resetView() {
    if (isUpdating) return;
    isUpdating = true;
    try {
        // 소분류 공간 하이라이트 필터링 정보 전면 롤백
        activeSubHighlight = null;
        const allSubItems = document.querySelectorAll('.legend-sub-item');
        allSubItems.forEach(item => item.classList.remove('active'));

        activeSearchIndices = null;
        lastHighlightedIndex = null;
        applyGraphVisualState();

        document.getElementById('info-panel').classList.remove('active');
        document.getElementById('search-input').value = '';
    } catch (e) { console.error('Reset error:', e); }
    finally { setTimeout(() => { isUpdating = false; }, 100); }
}

init();
