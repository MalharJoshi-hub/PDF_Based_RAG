// ================= PDF-RAG Studio Client Logic =================

let timerInterval = null;
let currentAnswer = "";
let currentCitations = [];

// ----------------- Tab Navigation -----------------
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active', 'border-indigo-500', 'text-indigo-400');
    btn.classList.add('border-transparent', 'text-slate-400');
  });

  const targetTab = document.getElementById('tab-' + tabId);
  const targetBtn = document.getElementById('tab-btn-' + tabId);
  if (targetTab) targetTab.classList.remove('hidden');
  if (targetBtn) {
    targetBtn.classList.add('active', 'border-indigo-500', 'text-indigo-400');
    targetBtn.classList.remove('border-transparent', 'text-slate-400');
  }

  if (tabId === 'corpus') {
    loadPDFs();
    loadEDA();
  }
}

// ----------------- Helper API -----------------
async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `HTTP error ${res.status}`);
  }
  return data;
}

// ----------------- Markdown / Text Formatter -----------------
function formatAnswerText(text) {
  if (!text) return "";
  // Bold formatting: **text** -> <strong class="text-white font-semibold">text</strong>
  let html = text.replace(/\*\*(.*?)\*\*/g, '<strong class="text-indigo-200 font-semibold">$1</strong>');
  
  // Bullet formatting: • bullet -> <li class="ml-4 list-disc text-slate-200 my-1">bullet</li>
  const lines = html.split('\n');
  const formattedLines = lines.map(line => {
    const trimmed = line.trim();
    if (trimmed.startsWith('•')) {
      return `<div class="flex items-start space-x-2 my-1.5"><span class="text-indigo-400 font-bold">•</span><span class="text-slate-200">${trimmed.substring(1).trim()}</span></div>`;
    }
    return line ? `<p class="my-1.5">${line}</p>` : '';
  });
  
  return formattedLines.join('');
}

// ----------------- Ask Documents Logic -----------------
function setQuery(text) {
  document.getElementById('query-input').value = text;
  document.getElementById('query-input').focus();
}

async function askQuestion() {
  const q = document.getElementById('query-input').value.trim();
  if (!q) {
    alert('Please enter a question.');
    return;
  }

  const retrieverSelect = document.getElementById('retriever-select');
  const retriever = retrieverSelect ? retrieverSelect.value : 'dense';
  const rerankToggle = document.getElementById('rerank-toggle');
  const useReranker = rerankToggle ? rerankToggle.checked : true;

  // UI state: loading
  const answerCard = document.getElementById('answer-card');
  if (answerCard) answerCard.classList.remove('hidden');
  
  const qaEmpty = document.getElementById('qa-empty');
  if (qaEmpty) qaEmpty.classList.add('hidden');
  
  const qaResult = document.getElementById('qa-result');
  if (qaResult) qaResult.classList.add('hidden');
  
  const timingBadge = document.getElementById('timing-badge');
  if (timingBadge) timingBadge.classList.add('hidden');
  
  const excerptsContainer = document.getElementById('excerpts-container');
  if (excerptsContainer) excerptsContainer.classList.add('hidden');
  
  const qaLoading = document.getElementById('qa-loading');
  if (qaLoading) qaLoading.classList.remove('hidden');

  const btn = document.getElementById('ask-btn');
  if (btn) {
    btn.disabled = true;
    btn.classList.add('opacity-75');
  }

  const startTime = Date.now();
  timerInterval = setInterval(() => {
    const sec = ((Date.now() - startTime) / 1000).toFixed(1);
    document.getElementById('loading-timer').innerText = `Reading documents & synthesizing answer (${sec}s)...`;
  }, 100);

  try {
    const payload = {
      query: q,
      retriever: retriever,
      use_reranker: useReranker,
      top_k_retrieved: 4,
      top_k_rerank: 2
    };

    const data = await api('/api/query', 'POST', payload);

    clearInterval(timerInterval);
    if (qaLoading) qaLoading.classList.add('hidden');

    currentAnswer = data.answer || "No information found in the uploaded documents.";
    
    // In our new light theme, we don't need custom bullet HTML to have text-slate-200.
    // Let's just set the innerHTML or innerText directly or use a lighter formatter.
    const answerTextEl = document.getElementById('answer-text');
    if (answerTextEl) {
      // Just render as raw text with whitespace-pre-line, but bold tags processed
      let html = currentAnswer.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      answerTextEl.innerHTML = html;
    }

    // Timings
    const t = data.timings || {};
    const totalSec = ((t.total_ms || 0) / 1000).toFixed(1);
    if (timingBadge) {
      timingBadge.innerText = `⚡ ${totalSec}s total`;
      timingBadge.classList.remove('hidden');
    }

    // Compact Source Chips
    currentCitations = data.citations || [];
    const chipsContainer = document.getElementById('source-chips');
    const excerptsContainer = document.getElementById('excerpts-container');

    if (currentCitations.length > 0) {
      if (chipsContainer) {
        chipsContainer.innerHTML = currentCitations.map(c => `
          <span class="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-surface-base border border-surface-border text-xs text-brand-text">
            <svg class="w-3.5 h-3.5 text-brand-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
            <span class="font-medium truncate max-w-[200px]" title="${c.filename}">${c.filename}</span>
            <span class="text-gray-500">&bull; P.${c.page_number}</span>
          </span>
        `).join('');
      }

      const sourcesBar = document.getElementById('sources-bar');
      if (sourcesBar) sourcesBar.classList.remove('hidden');
    } else {
      if (chipsContainer) chipsContainer.innerHTML = `<span class="text-xs text-gray-500 italic">No direct sources found</span>`;
      const sourcesBar = document.getElementById('sources-bar');
      if (sourcesBar) sourcesBar.classList.add('hidden');
    }

    if (qaResult) qaResult.classList.remove('hidden');

  } catch (err) {
    clearInterval(timerInterval);
    if (qaLoading) qaLoading.classList.add('hidden');
    alert(`Query Error: ${err.message}`);
    if (qaEmpty) qaEmpty.classList.remove('hidden');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.classList.remove('opacity-75');
    }
  }
}

function toggleExcerpts() {
  const container = document.getElementById('excerpts-container');
  const btn = document.getElementById('toggle-excerpts-btn');
  const isHidden = container.classList.contains('hidden');

  if (isHidden) {
    container.classList.remove('hidden');
    btn.innerHTML = `<span>Hide Source Excerpts</span><svg class="w-3.5 h-3.5 rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>`;
  } else {
    container.classList.add('hidden');
    btn.innerHTML = `<span>View Source Excerpts</span><svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>`;
  }
}

function expandExcerpt(idx) {
  const c = currentCitations[idx];
  if (!c) return;
  const p = document.getElementById(`excerpt-p-${idx}`);
  const btn = document.getElementById(`expand-btn-${idx}`);
  if (p && btn) {
    p.innerText = `"${c.full_text}"`;
    btn.style.display = 'none';
  }
}

function copyAnswer() {
  if (!currentAnswer) return;
  navigator.clipboard.writeText(currentAnswer).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.classList.add('text-emerald-400');
    setTimeout(() => btn.classList.remove('text-emerald-400'), 1500);
  });
}

// ----------------- PDF Document Management -----------------
async function loadPDFs() {
  try {
    const pdfs = await api('/api/pdfs');
    
    let totalChunks = 0;
    pdfs.forEach(p => totalChunks += (p.total_chunks || 0));
    document.getElementById('header-chunk-count').innerText = `${totalChunks} chunks`;

    const fullTable = document.getElementById('full-doc-table');
    if (fullTable) {
      if (!pdfs.length) {
        fullTable.innerHTML = `<tr><td colspan="4" class="py-6 text-center text-slate-500">No documents found. Upload a PDF using the box above.</td></tr>`;
      } else {
        fullTable.innerHTML = pdfs.map(p => `
          <tr class="hover:bg-surface-hover/30 transition">
            <td class="py-3 font-medium text-slate-200 truncate max-w-xs" title="${p.filename}">${p.filename}</td>
            <td class="py-3 text-slate-400 font-mono">${p.total_pages || 1}</td>
            <td class="py-3 text-indigo-400 font-mono font-semibold">${p.total_chunks || 0}</td>
            <td class="py-3 text-right">
              <button onclick="deletePDF(${p.id}, '${p.filename}')" class="text-rose-400 hover:text-rose-300 transition text-xs">
                Delete
              </button>
            </td>
          </tr>
        `).join('');
      }
    }
  } catch (err) {
    console.error('Failed to load PDFs:', err);
  }
}

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    alert('Only .pdf files are supported.');
    return;
  }

  const statusEl = document.getElementById('upload-status');
  statusEl.className = 'text-xs py-2 px-3 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 block';
  statusEl.innerText = `Processing and chunking ${file.name}...`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/pdfs', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');

    statusEl.className = 'text-xs py-2 px-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 block';
    statusEl.innerText = `Uploaded: ${data.filename} (${data.total_chunks} chunks indexed)`;
    await loadPDFs();
    setTimeout(() => { statusEl.classList.add('hidden'); }, 4000);
  } catch (err) {
    statusEl.className = 'text-xs py-2 px-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 block';
    statusEl.innerText = `Error: ${err.message}`;
  }
}

async function deletePDF(id, name) {
  if (!confirm(`Delete ${name} and rebuild all indices?`)) return;
  try {
    await api(`/api/pdfs/${id}`, 'DELETE');
    await loadPDFs();
  } catch (err) {
    alert(`Failed to delete: ${err.message}`);
  }
}

// ----------------- Benchmark Logic -----------------
async function runBenchmark() {
  const q = document.getElementById('compare-query-input').value.trim() || document.getElementById('query-input').value.trim();
  if (!q) {
    alert('Please enter a query to benchmark.');
    return;
  }

  const btn = document.getElementById('benchmark-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></span><span>Running...</span>`;

  try {
    const data = await api('/api/compare', 'POST', { query: q, top_k: 3 });

    document.getElementById('bench-dense-latency').innerText = `${data.dense.latency_ms} ms`;
    document.getElementById('bench-bm25-latency').innerText = `${data.bm25.latency_ms} ms`;
    document.getElementById('bench-tfidf-latency').innerText = `${data.tfidf.latency_ms} ms`;

    const comp = data.comparison || {};
    document.getElementById('overlap-dense-bm25').innerText = `${comp.overlap_dense_bm25_count} of 3`;
    document.getElementById('overlap-dense-tfidf').innerText = `${comp.overlap_dense_tfidf_count} of 3`;
    document.getElementById('overlap-all').innerText = `${comp.overlap_all_count} shared`;
    document.getElementById('benchmark-overlap-pills').classList.remove('hidden');

    const renderHits = (hits, elId, badgeClass) => {
      const container = document.getElementById(elId);
      if (!hits.length) {
        container.innerHTML = `<div class="text-slate-500 italic py-4 text-center">No hits.</div>`;
        return;
      }
      container.innerHTML = hits.map((h, i) => `
        <div class="p-3 rounded-xl bg-surface-base border border-surface-border text-xs space-y-1.5">
          <div class="flex items-center justify-between">
            <span class="font-medium text-slate-300 truncate max-w-[160px]">#${i + 1} ${h.filename || 'Doc'}</span>
            <span class="text-[10px] font-mono px-1.5 py-0.5 rounded ${badgeClass}">Score: ${h.score}</span>
          </div>
          <p class="text-slate-400 line-clamp-2 leading-relaxed">
            ${h.text || ''}
          </p>
        </div>
      `).join('');
    };

    renderHits(data.dense.results, 'bench-dense-hits', 'bg-indigo-500/10 text-indigo-400');
    renderHits(data.bm25.results, 'bench-bm25-hits', 'bg-purple-500/10 text-purple-400');
    renderHits(data.tfidf.results, 'bench-tfidf-hits', 'bg-emerald-500/10 text-emerald-400');

  } catch (err) {
    alert(`Benchmark error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg><span>Run Benchmark</span>`;
  }
}

// ----------------- EDA Dashboard -----------------
async function loadEDA() {
  try {
    const data = await api('/api/eda');
    document.getElementById('eda-total-chunks').innerText = data.total_chunks || 0;
    document.getElementById('eda-total-words').innerText = data.total_words ? data.total_words.toLocaleString() : 0;
    document.getElementById('eda-unique-words').innerText = data.total_unique_words ? data.total_unique_words.toLocaleString() : 0;
    document.getElementById('eda-mean-chunk').innerText = data.mean_chunk_words || 0;
    document.getElementById('eda-median-chunk').innerText = data.median_chunk_words || 0;
    document.getElementById('eda-std-chunk').innerText = data.std_chunk_words || 0;

    const ts = Date.now();
    document.getElementById('chart-chunk-len').src = `/static/eda/chunk_length_distribution.png?t=${ts}`;
    document.getElementById('chart-keywords').src = `/static/eda/word_frequency.png?t=${ts}`;
    document.getElementById('chart-wordcloud').src = `/static/eda/wordcloud.png?t=${ts}`;
  } catch (err) {
    console.log('EDA loading:', err);
  }
}

async function runEDAServer() {
  const btn = document.getElementById('eda-run-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></span><span>Computing...</span>`;

  try {
    await api('/api/eda/run', 'POST');
    await loadEDA();
  } catch (err) {
    alert(`EDA error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg><span>Re-compute EDA</span>`;
  }
}

// ----------------- Drag & Drop -----------------
function setupDragAndDrop() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  if (!dropZone || !fileInput) return;

  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) uploadFile(e.target.files[0]);
  });

  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('border-indigo-500', 'bg-indigo-500/5');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('border-indigo-500', 'bg-indigo-500/5');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length) uploadFile(files[0]);
  });
}

// ----------------- Keyboard Shortcut -----------------
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    const qaTab = document.getElementById('tab-qa');
    if (!qaTab.classList.contains('hidden')) {
      askQuestion();
    }
  }
});

// ----------------- Init -----------------
window.addEventListener('DOMContentLoaded', () => {
  loadPDFs();
  setupDragAndDrop();
});
