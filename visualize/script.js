let globalNodes = [];
let globalEdges = [];
let originalEdgeX = [], originalEdgeY = [], originalEdgeZ = [];
let isUpdating = false;

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

        globalEdges.forEach(edge => {
            const s = globalNodes[edge[0]], t = globalNodes[edge[1]];
            originalEdgeX.push(s.x, t.x, null); originalEdgeY.push(s.y, t.y, null); originalEdgeZ.push(s.z, t.z, null);
        });
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
        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            if (!query) { resetView(); return; }
            const matches = globalNodes.map(n => {
                const title = (n.metadata.title || "").toLowerCase();
                const tags = (n.metadata.tags || []).join(" ").toLowerCase();
                const text = (n.text || "").toLowerCase();
                return title.includes(query) || tags.includes(query) || text.includes(query);
            });
            Plotly.restyle('plot', {
                'marker.opacity': [matches.map(m => m ? 1.0 : 0.05)],
                'marker.size': [matches.map((m, i) => m ? globalNodes[i].size * 1.5 : 2)]
            }, [1]);
            Plotly.restyle('plot', { 'line.color': 'rgba(0,0,0,0)' }, [0]);
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
    if (isUpdating) return;
    isUpdating = true;
    try {
        const item = globalNodes[index];
        if (!item) return;

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

        processedContent = processedContent.replace(/^\[(#{1,6}.+?)\]/gm, '$1');
        // --- Math Protection: Hide math from marked.js ---
        const mathBlocks = [];
        // Protect display math ($$, \[...\]) and inline math ($, \(...\))
        processedContent = processedContent.replace(/\$\$(.*?)\$\$|\$(.*?)\$|\\\[(.*?)\\\]|\\\((.*?)\\\)/gs, (match) => {
            const id = `@@MATH${mathBlocks.length}@@`;
            mathBlocks.push(match);
            return id;
        });

        const infoContent = document.getElementById('info-content');
        let html = marked.parse(processedContent.substring(0, 5000));
        
        // --- Math Restoration: Restore math tokens ---
        html = html.replace(/@@MATH(\d+)@@/g, (match, id) => {
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
            if (edge[0] === index || edge[1] === index) {
                connectedIndices.add(edge[0] === index ? edge[1] : edge[0]);
                const s = globalNodes[edge[0]], t = globalNodes[edge[1]];
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
    } catch (e) { console.error('Reset error:', e); }
    finally { setTimeout(() => { isUpdating = false; }, 100); }
}

init();
