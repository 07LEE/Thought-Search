let globalNodes = [];
let globalEdges = [];
let originalEdgeX = [], originalEdgeY = [], originalEdgeZ = [];
let isUpdating = false;
let currentThreshold = 0.80;
let lastHighlightedIndex = null;

async function init() {
    try {
        marked.setOptions({ gfm: true, breaks: true, headerIds: false, mangle: false });

        const response = await fetch('/data/viz-data.json');
        const data = await response.json();
        globalNodes = data.nodes;
        globalEdges = data.edges;
        const categories = data.categories;

        const legendContainer = document.getElementById('legend');
        legendContainer.innerHTML = '<div style="font-size: 0.7rem; color: #64748b; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">Categories</div>';

        for (const [cat, color] of Object.entries(categories)) {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `<div class="legend-color" style="background: ${color}"></div><span>${cat}</span>`;
            legendContainer.appendChild(item);
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


        const getFilteredEdges = (threshold) => {
            const edgeX = [], edgeY = [], edgeZ = [];
            globalEdges.forEach(edge => {
                const [sourceIdx, targetIdx, score] = edge;
                if (score >= threshold) {
                    const s = globalNodes[sourceIdx], t = globalNodes[targetIdx];
                    edgeX.push(s.x, t.x, null); edgeY.push(s.y, t.y, null); edgeZ.push(s.z, t.z, null);
                }
            });
            return { x: edgeX, y: edgeY, z: edgeZ };
        };

        const initialEdges = getFilteredEdges(currentThreshold);
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
            
            if (lastHighlightedIndex !== null) {
                // If in highlight mode, re-trigger highlight with new threshold
                highlightNode(lastHighlightedIndex);
            } else {
                // Normal mode: update all edges
                const newEdges = getFilteredEdges(currentThreshold);
                originalEdgeX = newEdges.x;
                originalEdgeY = newEdges.y;
                originalEdgeZ = newEdges.z;
                
                Plotly.restyle('plot', {
                    'x': [originalEdgeX],
                    'y': [originalEdgeY],
                    'z': [originalEdgeZ]
                }, [0]);
            }
        });

        const plotDiv = document.getElementById('plot');
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

        await Plotly.restyle('plot', {
            'marker.opacity': [globalNodes.map((n, i) => (i === index || connectedIndices.has(i)) ? 1.0 : 0.05)],
            'marker.size': [globalNodes.map((n, i) => (i === index || connectedIndices.has(i)) ? n.size * 1.5 : 2)]
        }, [1]);

        await Plotly.restyle('plot', {
            'x': [hX.length ? hX : [null]], 'y': [hY.length ? hY : [null]], 'z': [hZ.length ? hZ : [null]],
            'line.color': 'rgba(56, 189, 248, 0.8)', 'line.width': 3
        }, [0]);

        document.getElementById('info-panel').classList.add('active');
    } catch (e) { console.error('Highlight error:', e); }
    finally { setTimeout(() => { isUpdating = false; }, 100); }
}

async function resetView() {
    if (isUpdating) return;
    isUpdating = true;
    try {
        await Plotly.restyle('plot', {
            'marker.opacity': [globalNodes.map(() => 0.9)],
            'marker.size': [globalNodes.map(n => n.size)],
            'marker.color': [globalNodes.map(n => n.color)]
        }, [1]);

        await Plotly.restyle('plot', {
            'x': [originalEdgeX], 'y': [originalEdgeY], 'z': [originalEdgeZ],
            'line.color': 'rgba(56, 189, 248, 0.15)', 'line.width': 1
        }, [0]);

        document.getElementById('info-panel').classList.remove('active');
        document.getElementById('search-input').value = '';
        lastHighlightedIndex = null;
    } catch (e) { console.error('Reset error:', e); }
    finally { setTimeout(() => { isUpdating = false; }, 100); }
}

init();
