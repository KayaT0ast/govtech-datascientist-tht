/* ── RAG Evaluation Platform — SPA ── */

const API = '';   // same-origin; prefix with http://localhost:8000 if serving separately

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  token:           localStorage.getItem('rag_token') || null,
  username:        localStorage.getItem('rag_user')  || null,
  currentSystem:   localStorage.getItem('rag_system') || '',
  currentPage:     'dashboard',
  evalSystem:      'strict',     // active system tab on results page
  chartInstances:  {},
};

// ── API helpers ────────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
  if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
  const res = await fetch(API + path, { ...opts, headers });
  if (res.status === 401) { logout(); return null; }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function apiPost(path, body) {
  return apiFetch(path, { method: 'POST', body: JSON.stringify(body) });
}

async function apiDelete(path, body) {
  return apiFetch(path, { method: 'DELETE', body: JSON.stringify(body) });
}

// ── Auth ───────────────────────────────────────────────────────────────────────
function logout() {
  localStorage.removeItem('rag_token');
  localStorage.removeItem('rag_user');
  state.token = null;
  state.username = null;
  renderApp();
}

// ── Router ─────────────────────────────────────────────────────────────────────
function navigate(page) {
  state.currentPage = page;
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.page === page);
  });
  renderPage(page);
}

// ── Main render ────────────────────────────────────────────────────────────────
function renderApp() {
  if (!state.token) {
    document.getElementById('app').innerHTML = renderAuthPage();
    bindAuthPage();
    return;
  }
  document.getElementById('app').innerHTML = renderShell();
  bindShell();
  navigate(state.currentPage);
}

function renderShell() {
  return `
    <nav id="navbar">
      <div class="nav-logo">
        <span class="nav-logo-badge">RAG</span>
        <span>Evaluation Platform</span>
      </div>
      <div class="nav-spacer"></div>
      <div class="nav-user">
        <span class="text-sm text-muted">${escHtml(state.username)}</span>
        <div class="nav-avatar">${(state.username || '?')[0].toUpperCase()}</div>
        <button class="btn btn-ghost btn-sm" onclick="logout()">Sign out</button>
      </div>
    </nav>

    <aside id="sidebar">
      <div class="nav-section-title">Overview</div>
      <button class="nav-item" data-page="dashboard" onclick="navigate('dashboard')">
        ${icon('grid')} Dashboard
      </button>

      <div class="nav-section-title">Workspace</div>
      <button class="nav-item" data-page="setup" onclick="navigate('setup')">
        ${icon('upload')} Upload KB
      </button>
      <button class="nav-item" data-page="questionnaire" onclick="navigate('questionnaire')">
        ${icon('list')} Questionnaire
      </button>
      <button class="nav-item" data-page="evaluate" onclick="navigate('evaluate')">
        ${icon('play')} Evaluate
      </button>

      <div class="nav-section-title">Results</div>
      <button class="nav-item" data-page="results" onclick="navigate('results')">
        ${icon('bar-chart')} Metrics
      </button>
      <button class="nav-item" data-page="insights" onclick="navigate('insights')">
        ${icon('lightbulb')} Insights
      </button>

      <div class="nav-section-title">Tools</div>
      <button class="nav-item" data-page="demo" onclick="navigate('demo')">
        ${icon('message')} Demo RAG
      </button>
    </aside>

    <main id="main">
      <div id="page-container"></div>
    </main>

    <div id="modal-container"></div>
  `;
}

function renderPage(page) {
  const container = document.getElementById('page-container');
  const renderers = {
    dashboard:     renderDashboard,
    setup:         renderSetup,
    questionnaire: renderQuestionnaire,
    evaluate:      renderEvaluate,
    results:       renderResults,
    insights:      renderInsights,
    demo:          renderDemo,
  };
  if (renderers[page]) {
    container.innerHTML = '<div class="page">' + renderers[page]() + '</div>';
    const binders = {
      dashboard:     bindDashboard,
      setup:         bindSetup,
      questionnaire: bindQuestionnaire,
      evaluate:      bindEvaluate,
      results:       bindResults,
      insights:      bindInsights,
      demo:          bindDemo,
    };
    if (binders[page]) binders[page]();
  }
}

// ── Auth page ──────────────────────────────────────────────────────────────────
function renderAuthPage() {
  return `
    <div id="auth-page">
      <div class="auth-left">
        <div class="auth-logo-wrap">
          <span class="auth-logo-text">RAG Evaluation Platform</span>
          <span class="auth-logo-badge">BETA</span>
        </div>
        <h1 class="auth-tagline">Benchmark Your RAG System</h1>
        <p class="auth-sub">Upload your knowledge base, connect your RAG system, and get deep diagnostic insights on retrieval and generation quality.</p>
      </div>
      <div class="auth-right">
        <div class="auth-box">
          <div class="auth-tabs">
            <button class="auth-tab active" id="tab-login" onclick="switchAuthTab('login')">Sign In</button>
            <button class="auth-tab" id="tab-register" onclick="switchAuthTab('register')">Register</button>
          </div>
          <div id="auth-error" class="alert alert-error mb-4 hidden"></div>
          <div id="auth-login-form">
            <div class="form-group mb-4">
              <label class="form-label">Username</label>
              <input class="form-input" id="login-user" type="text" placeholder="your username" autocomplete="username" />
            </div>
            <div class="form-group mb-4">
              <label class="form-label">Password</label>
              <input class="form-input" id="login-pass" type="password" placeholder="••••••••" autocomplete="current-password" />
            </div>
            <button class="btn btn-primary w-full" id="btn-login" onclick="doLogin()">Sign In</button>
          </div>
          <div id="auth-register-form" class="hidden">
            <div class="form-group mb-4">
              <label class="form-label">Username</label>
              <input class="form-input" id="reg-user" type="text" placeholder="choose a username" />
            </div>
            <div class="form-group mb-4">
              <label class="form-label">Password</label>
              <input class="form-input" id="reg-pass" type="password" placeholder="••••••••" />
            </div>
            <button class="btn btn-primary w-full" id="btn-register" onclick="doRegister()">Create Account</button>
          </div>
        </div>
      </div>
    </div>
  `;
}

function bindAuthPage() {
  const inputs = ['login-user', 'login-pass', 'reg-user', 'reg-pass'];
  inputs.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('keydown', e => { if (e.key === 'Enter') e.target.closest('[id$="-form"]').querySelector('button').click(); });
  });
}

function switchAuthTab(tab) {
  document.getElementById('tab-login').classList.toggle('active', tab === 'login');
  document.getElementById('tab-register').classList.toggle('active', tab === 'register');
  document.getElementById('auth-login-form').classList.toggle('hidden', tab !== 'login');
  document.getElementById('auth-register-form').classList.toggle('hidden', tab !== 'register');
  document.getElementById('auth-error').classList.add('hidden');
}

async function doLogin() {
  const user = document.getElementById('login-user').value.trim();
  const pass = document.getElementById('login-pass').value;
  const btn  = document.getElementById('btn-login');
  showAuthError('');
  btn.disabled = true;
  btn.textContent = 'Signing in…';
  try {
    const data = await apiPost('/api/auth/login', { username: user, password: pass });
    if (!data) return;
    state.token    = data.token;
    state.username = data.username;
    localStorage.setItem('rag_token', data.token);
    localStorage.setItem('rag_user', data.username);
    renderApp();
  } catch (e) {
    showAuthError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Sign In';
  }
}

async function doRegister() {
  const user = document.getElementById('reg-user').value.trim();
  const pass = document.getElementById('reg-pass').value;
  const btn  = document.getElementById('btn-register');
  showAuthError('');
  btn.disabled = true;
  btn.textContent = 'Creating…';
  try {
    const data = await apiPost('/api/auth/register', { username: user, password: pass });
    if (!data) return;
    state.token    = data.token;
    state.username = data.username;
    localStorage.setItem('rag_token', data.token);
    localStorage.setItem('rag_user', data.username);
    renderApp();
  } catch (e) {
    showAuthError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Create Account';
  }
}

function showAuthError(msg) {
  const el = document.getElementById('auth-error');
  if (!el) return;
  if (msg) { el.textContent = msg; el.classList.remove('hidden'); }
  else     { el.classList.add('hidden'); }
}

// ── Dashboard ──────────────────────────────────────────────────────────────────
function renderDashboard() {
  return `
    <div class="page-header">
      <div class="section-label">Overview</div>
      <h1 class="page-title">Dashboard</h1>
    </div>
    <div class="hero-card">
      <div class="hero-title">Welcome back, ${escHtml(state.username)} 👋</div>
      <div class="hero-sub">
        ${state.currentSystem
          ? `Active system: <strong>${escHtml(state.currentSystem)}</strong>`
          : 'No active system — upload a knowledge base to get started.'}
      </div>
      <div class="hero-actions">
        <button class="btn-hero" onclick="navigate('setup')">${icon('upload')} Upload KB</button>
        <button class="btn-hero" onclick="navigate('evaluate')">${icon('play')} Run Evaluation</button>
        <button class="btn-hero" onclick="navigate('results')">${icon('bar-chart')} View Results</button>
      </div>
    </div>

    <div class="stats-row" id="dash-stats">
      <div class="stat-card">
        <div class="stat-label">Chunks indexed</div>
        <div class="stat-value" id="stat-chunks">—</div>
        <div class="stat-meta">Knowledge base size</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Questions</div>
        <div class="stat-value" id="stat-questions">—</div>
        <div class="stat-meta">In questionnaire</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">E2E-OOKRI</div>
        <div class="stat-value" id="stat-ookri">—</div>
        <div class="stat-meta">Latest composite score</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Evaluation runs</div>
        <div class="stat-value" id="stat-runs">—</div>
        <div class="stat-meta">Total runs for system</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h3>RAG Systems</h3>
        <button class="btn btn-outline btn-sm" onclick="navigate('setup')">+ New System</button>
      </div>
      <div class="card-body" id="systems-list">
        <div class="flex items-center gap-2"><div class="spinner"></div> <span class="text-muted text-sm">Loading…</span></div>
      </div>
    </div>
  `;
}

async function bindDashboard() {
  // Load systems
  try {
    const data = await apiFetch('/api/systems');
    if (!data) return;
    const list = document.getElementById('systems-list');
    if (!data.systems || data.systems.length === 0) {
      list.innerHTML = `<p class="text-muted text-sm">No systems yet. <button class="btn btn-primary btn-sm" onclick="navigate('setup')">Upload a knowledge base</button></p>`;
    } else {
      list.innerHTML = data.systems.map(s => `
        <div class="flex items-center gap-4" style="padding:.5rem 0;border-bottom:1px solid var(--border);">
          <div style="flex:1;font-weight:600">${escHtml(s)}</div>
          <button class="btn btn-ghost btn-sm" onclick="setSystem('${escHtml(s)}')">
            ${state.currentSystem === s ? '<span class="badge badge-success">Active</span>' : 'Select'}
          </button>
        </div>`).join('');
    }
  } catch (e) {
    document.getElementById('systems-list').innerHTML = `<div class="alert alert-error">${e.message}</div>`;
  }

  // Load stats for current system
  if (!state.currentSystem) return;
  try {
    const [kb, q, runs] = await Promise.all([
      apiFetch(`/api/kb/status?rag_system_name=${enc(state.currentSystem)}`),
      apiFetch(`/api/questionnaire?rag_system_name=${enc(state.currentSystem)}`),
      apiFetch(`/api/runs?rag_system_name=${enc(state.currentSystem)}&system=strict`),
    ]);
    if (kb) setText('stat-chunks', kb.chunk_count);
    if (q)  setText('stat-questions', q.count);
    if (runs) {
      setText('stat-runs', runs.runs.length);
      if (runs.runs.length) {
        const m = await apiFetch(`/api/metrics?rag_system_name=${enc(state.currentSystem)}&system=strict`);
        if (m && m.metrics && m.metrics.e2e) {
          const ookri = m.metrics.e2e.E2E_OOKRI;
          setText('stat-ookri', typeof ookri === 'number' ? ookri.toFixed(3) : '—');
        }
      }
    }
  } catch (_) {}
}

function setSystem(name) {
  state.currentSystem = name;
  localStorage.setItem('rag_system', name);
  navigate('dashboard');
}

// ── Setup page ─────────────────────────────────────────────────────────────────
let _uploadFiles = [];

function renderSetup() {
  return `
    <div class="page-header">
      <div class="section-label">Workspace</div>
      <h1 class="page-title">Upload Knowledge Base</h1>
      <p class="page-desc">Upload your documents and set a system name to start the evaluation pipeline.</p>
    </div>

    <div class="card mb-4">
      <div class="card-header"><h3>System Name</h3></div>
      <div class="card-body">
        <div class="form-group">
          <label class="form-label">RAG System Name</label>
          <input class="form-input" id="setup-sys-name" type="text"
            placeholder="e.g. my-rag-system"
            value="${escHtml(state.currentSystem)}" />
          <div class="form-hint">Lowercase letters, numbers, hyphens and underscores only.</div>
        </div>
      </div>
    </div>

    <div class="card mb-4">
      <div class="card-header"><h3>Documents</h3></div>
      <div class="card-body">
        <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
          <div class="dropzone-icon">📄</div>
          <div class="dropzone-label"><strong>Click to browse</strong> or drag files here</div>
          <div class="dropzone-label text-xs mt-2">Supported: PDF, DOCX, PPTX, TXT, MD</div>
        </div>
        <input id="file-input" type="file" multiple accept=".pdf,.docx,.pptx,.txt,.md" class="hidden" />
        <div class="file-list" id="file-list"></div>
      </div>
    </div>

    <div class="card mb-4">
      <div class="card-header"><h3>Options</h3></div>
      <div class="card-body">
        <div class="toggle-wrap mb-4">
          <label class="toggle"><input type="checkbox" id="setup-reindex" /><span class="toggle-slider"></span></label>
          <span class="text-sm font-semibold">Re-index (delete existing chunks, re-embed and regenerate questionnaire)</span>
        </div>
        <div class="form-group">
          <label class="form-label">
            Questionnaire Scale
            <span class="alpha-display" id="scale-display">40</span>
          </label>
          <input type="range" id="setup-scale" min="10" max="100" step="10" value="40"
            oninput="document.getElementById('scale-display').textContent = this.value" />
          <div class="alpha-labels">
            <span>Quick (10)</span>
            <span>Standard (40)</span>
            <span>Thorough (100)</span>
          </div>
          <div class="form-hint">Number of evaluation questions to generate from your knowledge base.</div>
        </div>
      </div>
    </div>

    <div id="setup-alert" class="hidden mb-4"></div>

    <div class="flex items-center gap-4">
      <button class="btn btn-primary btn-lg" id="btn-upload" onclick="doUpload()">
        ${icon('upload')} Upload &amp; Generate Questionnaire
      </button>
      <span id="upload-status" class="text-muted text-sm"></span>
    </div>

    <div id="setup-job-area" class="hidden mt-6"></div>
  `;
}

function bindSetup() {
  _uploadFiles = [];
  const dropzone  = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  fileInput.addEventListener('change', () => addFiles(Array.from(fileInput.files)));
  dropzone.addEventListener('dragover',  e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    addFiles(Array.from(e.dataTransfer.files));
  });
}

function addFiles(files) {
  files.forEach(f => {
    if (!_uploadFiles.find(x => x.name === f.name)) _uploadFiles.push(f);
  });
  renderFileList();
}

function removeFile(name) {
  _uploadFiles = _uploadFiles.filter(f => f.name !== name);
  renderFileList();
}

function renderFileList() {
  const list = document.getElementById('file-list');
  if (!list) return;
  list.innerHTML = _uploadFiles.map(f => `
    <div class="file-item">
      <span>📄</span>
      <span class="file-item-name">${escHtml(f.name)}</span>
      <span class="text-xs text-muted">${(f.size / 1024).toFixed(0)} KB</span>
      <button class="file-remove" onclick="removeFile('${escHtml(f.name)}')" title="Remove">×</button>
    </div>`).join('');
}

async function doUpload() {
  const sysName = document.getElementById('setup-sys-name').value.trim();
  const reindex = document.getElementById('setup-reindex').checked;
  const scale   = parseInt(document.getElementById('setup-scale').value, 10) || 40;
  const alertEl = document.getElementById('setup-alert');

  if (!sysName) { showAlert(alertEl, 'error', 'Enter a system name.'); return; }
  if (_uploadFiles.length === 0) { showAlert(alertEl, 'error', 'Select at least one document.'); return; }

  alertEl.classList.add('hidden');
  const btn = document.getElementById('btn-upload');
  btn.disabled = true;

  const form = new FormData();
  form.append('rag_system_name', sysName);
  form.append('reindex', reindex ? 'true' : 'false');
  form.append('scale', scale);
  _uploadFiles.forEach(f => form.append('files', f));

  try {
    const headers = {};
    if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
    const res = await fetch(API + '/api/kb/upload', { method: 'POST', body: form, headers });
    if (!res.ok) { const b = await res.json().catch(() => {}); throw new Error(b?.detail || `HTTP ${res.status}`); }
    const data = await res.json();
    state.currentSystem = sysName;
    localStorage.setItem('rag_system', sysName);
    pollJob(data.job_id, 'setup-job-area', (result) => {
      const qCount = result?.question_count ?? '?';
      showAlert(alertEl, 'success',
        `Knowledge base ready — ${result?.chunk_count ?? '?'} chunks embedded, ${qCount} questions generated. System "${sysName}" is ready to evaluate.`);
      _uploadFiles = [];
      renderFileList();
      btn.disabled = false;
    }, () => { btn.disabled = false; });
  } catch (e) {
    showAlert(alertEl, 'error', e.message);
    btn.disabled = false;
  }
}

// ── Questionnaire page ─────────────────────────────────────────────────────────
function renderQuestionnaire() {
  const sys = state.currentSystem;
  return `
    <div class="page-header">
      <div class="section-label">Workspace</div>
      <h1 class="page-title">Questionnaire</h1>
      <p class="page-desc">View and edit the evaluation questions generated from your knowledge base. Changes are saved directly to MongoDB.</p>
    </div>

    <div class="flex items-center gap-4 mb-4" style="flex-wrap:wrap;">
      <div class="form-group" style="min-width:220px;">
        <label class="form-label">System</label>
        <input class="form-input" id="q-sys-name" value="${escHtml(sys)}" placeholder="rag-system-name" />
      </div>
      <div style="padding-top:1.4rem;">
        <button class="btn btn-primary" onclick="loadQuestionnaire()">Load</button>
      </div>
      <div style="padding-top:1.4rem;" id="q-count-wrap" class="hidden">
        <span class="badge badge-neutral" id="q-count-badge"></span>
      </div>
    </div>

    <div id="q-alert" class="hidden mb-4"></div>
    <div id="q-list"></div>
  `;
}

async function bindQuestionnaire() {
  if (state.currentSystem) await loadQuestionnaire();
}

async function loadQuestionnaire() {
  const sys = (document.getElementById('q-sys-name') || {}).value?.trim() || state.currentSystem;
  if (!sys) return;
  const listEl  = document.getElementById('q-list');
  const alertEl = document.getElementById('q-alert');
  listEl.innerHTML = '<div class="flex items-center gap-2 text-muted text-sm"><div class="spinner"></div> Loading questions…</div>';

  try {
    const data = await apiFetch(`/api/questionnaire?rag_system_name=${enc(sys)}`);
    if (!data) return;
    const badge = document.getElementById('q-count-badge');
    const wrap  = document.getElementById('q-count-wrap');
    if (badge) { badge.textContent = `${data.count} questions`; wrap.classList.remove('hidden'); }

    if (!data.questions || data.questions.length === 0) {
      listEl.innerHTML = '<div class="alert alert-info">No questions found. Upload a knowledge base to generate them.</div>';
      return;
    }
    listEl.innerHTML = data.questions.map((q, i) => renderQuestionCard(q, i, sys)).join('');
  } catch (e) {
    listEl.innerHTML = `<div class="alert alert-error">${escHtml(e.message)}</div>`;
  }
}

function renderQuestionCard(q, idx, sys) {
  const id = escHtml(q.id || '');
  const safeId = id.replace(/[^a-zA-Z0-9_]/g, '_');
  return `
    <div class="card mb-3" id="qcard_${safeId}">
      <div class="card-header" style="justify-content:space-between;align-items:center;">
        <div class="flex items-center gap-2">
          <span class="badge badge-neutral text-xs">#${idx + 1}</span>
          <span class="text-xs text-muted" style="font-family:monospace">${id}</span>
          <span class="badge ${q.type === 'OOKB' ? 'badge-warning' : 'badge-success'} text-xs">${escHtml(q.type || '')}</span>
          <span class="badge badge-neutral text-xs">${escHtml(q.category || '')}</span>
          <span class="badge badge-neutral text-xs">${escHtml(q.difficulty || '')}</span>
        </div>
        <div class="flex items-center gap-2">
          <button class="btn btn-ghost btn-sm" onclick="editQuestion('${safeId}')">Edit</button>
        </div>
      </div>
      <div class="card-body" id="qview_${safeId}">
        <div class="mb-3">
          <div class="form-label text-xs">Question</div>
          <div class="text-sm" style="white-space:pre-wrap">${escHtml(q.question || '')}</div>
        </div>
        <div class="flex gap-6">
          <div style="flex:1">
            <div class="form-label text-xs">Expected Behavior</div>
            <div class="text-sm">${escHtml(q.expected_behavior || '')}</div>
          </div>
          <div style="flex:2">
            <div class="form-label text-xs">Expected Answer</div>
            <div class="text-sm" style="white-space:pre-wrap">${escHtml(q.expected_answer || '')}</div>
          </div>
        </div>
      </div>
      <div class="card-body hidden" id="qedit_${safeId}">
        <div class="form-group mb-3">
          <label class="form-label">Question</label>
          <textarea class="form-input" id="qf_question_${safeId}" rows="3" style="resize:vertical">${escHtml(q.question || '')}</textarea>
        </div>
        <div class="url-row">
          <div class="form-group">
            <label class="form-label">Category</label>
            <select class="form-select" id="qf_category_${safeId}">
              ${['in_kb','out_of_kb','partial_kb','adversarial','temporal'].map(c =>
                `<option value="${c}" ${q.category === c ? 'selected' : ''}>${c}</option>`).join('')}
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Type</label>
            <select class="form-select" id="qf_type_${safeId}">
              <option value="IKB" ${q.type === 'IKB' ? 'selected' : ''}>IKB</option>
              <option value="OOKB" ${q.type === 'OOKB' ? 'selected' : ''}>OOKB</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Difficulty</label>
            <select class="form-select" id="qf_difficulty_${safeId}">
              ${['easy','medium','hard'].map(d =>
                `<option value="${d}" ${q.difficulty === d ? 'selected' : ''}>${d}</option>`).join('')}
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Expected Behavior</label>
            <select class="form-select" id="qf_expected_behavior_${safeId}">
              <option value="answer" ${q.expected_behavior === 'answer' ? 'selected' : ''}>answer</option>
              <option value="refuse" ${q.expected_behavior === 'refuse' ? 'selected' : ''}>refuse</option>
              <option value="partial" ${q.expected_behavior === 'partial' ? 'selected' : ''}>partial</option>
            </select>
          </div>
        </div>
        <div class="form-group mb-3">
          <label class="form-label">Expected Answer</label>
          <textarea class="form-input" id="qf_expected_answer_${safeId}" rows="3" style="resize:vertical">${escHtml(q.expected_answer || '')}</textarea>
        </div>
        <div class="flex items-center gap-3">
          <button class="btn btn-primary btn-sm" onclick="saveQuestion('${escHtml(q.id)}','${safeId}')">Save</button>
          <button class="btn btn-ghost btn-sm" onclick="cancelEditQuestion('${safeId}')">Cancel</button>
          <span class="text-xs text-muted hidden" id="qsave_status_${safeId}"></span>
        </div>
      </div>
    </div>
  `;
}

function editQuestion(safeId) {
  document.getElementById(`qview_${safeId}`).classList.add('hidden');
  document.getElementById(`qedit_${safeId}`).classList.remove('hidden');
}

function cancelEditQuestion(safeId) {
  document.getElementById(`qedit_${safeId}`).classList.add('hidden');
  document.getElementById(`qview_${safeId}`).classList.remove('hidden');
}

async function saveQuestion(qId, safeId) {
  const sys = (document.getElementById('q-sys-name') || {}).value?.trim() || state.currentSystem;
  const statusEl = document.getElementById(`qsave_status_${safeId}`);
  statusEl.textContent = 'Saving…';
  statusEl.classList.remove('hidden');

  const body = {
    rag_system_name:   sys,
    question:          document.getElementById(`qf_question_${safeId}`).value,
    category:          document.getElementById(`qf_category_${safeId}`).value,
    type:              document.getElementById(`qf_type_${safeId}`).value,
    difficulty:        document.getElementById(`qf_difficulty_${safeId}`).value,
    expected_behavior: document.getElementById(`qf_expected_behavior_${safeId}`).value,
    expected_answer:   document.getElementById(`qf_expected_answer_${safeId}`).value,
  };

  try {
    const data = await apiFetch(`/api/questionnaire/${enc(qId)}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    });
    if (!data) return;
    statusEl.textContent = 'Saved ✓';
    // Update the view panel with new values
    const viewEl = document.getElementById(`qview_${safeId}`);
    viewEl.querySelector('[style*="white-space:pre-wrap"]').textContent = body.question;
    cancelEditQuestion(safeId);
    setTimeout(() => { statusEl.textContent = ''; statusEl.classList.add('hidden'); }, 2000);
  } catch (e) {
    statusEl.textContent = 'Error: ' + e.message;
  }
}

// ── Evaluate page ──────────────────────────────────────────────────────────────
function renderEvaluate() {
  const sys = state.currentSystem;
  return `
    <div class="page-header">
      <div class="section-label">Workspace</div>
      <h1 class="page-title">Run Evaluation</h1>
      <p class="page-desc">Configure your RAG system endpoints and kick off the 3-mode evaluation pipeline.</p>
    </div>

    <div class="card mb-4">
      <div class="card-header"><h3>System</h3></div>
      <div class="card-body">
        <div class="form-group">
          <label class="form-label">Name of Evaluation Run</label>
          <input class="form-input" id="eval-sys-name" type="text" value="${escHtml(sys)}" placeholder="my-rag-system" />
        </div>
      </div>
    </div>

    <div class="card mb-4">
      <div class="card-header">
        <h3>External RAG System URLs</h3>
        <span class="badge badge-neutral text-xs">E2E URL required</span>
      </div>
      <div class="card-body">
        <div class="url-row">
          <div class="form-group">
            <label class="form-label">E2E RAG URL <span class="text-danger">*</span></label>
            <input class="form-input" id="eval-e2e-url" type="url" placeholder="https://your-rag-api.com/query" />
            <div class="form-hint">Must accept <code>{"query": "..."}</code> and return <code>{"response", "retrieved_contexts", "retrieved_ids"}</code></div>
          </div>
          <div class="form-group">
            <label class="form-label">Retriever URL <span class="text-muted">(optional)</span></label>
            <input class="form-input" id="eval-ret-url" type="url" placeholder="https://your-rag-api.com/retrieve" />
            <div class="form-hint">Accepts <code>{"query", "top_k"}</code>, returns <code>{"chunks": [...]}</code></div>
          </div>
        </div>
        <div class="url-row mt-4">
          <div class="form-group">
            <label class="form-label">Generator URL <span class="text-muted">(optional)</span></label>
            <input class="form-input" id="eval-gen-url" type="url" placeholder="https://your-rag-api.com/generate" />
            <div class="form-hint">Accepts <code>{"query", "context"}</code>, returns <code>{"response"}</code></div>
          </div>
          <div></div>
        </div>
      </div>
    </div>

    <div class="card mb-4">
      <div class="card-header"><h3>Parameters</h3></div>
      <div class="card-body">
        <div class="form-group mb-4">
          <label class="form-label">
            Deployment Weight (α)
            <span class="alpha-display" id="alpha-display">0.80</span>
          </label>
          <input type="range" id="alpha-slider" min="0" max="1" step="0.05" value="0.80"
            oninput="document.getElementById('alpha-display').textContent = parseFloat(this.value).toFixed(2)" />
          <div class="alpha-labels">
            <span>Internal-use (α=0.25)</span>
            <span>Balanced (α=0.50)</span>
            <span>Public-facing (α=0.80)</span>
          </div>
        </div>
        <div class="toggle-wrap">
          <label class="toggle"><input type="checkbox" id="eval-regen" /><span class="toggle-slider"></span></label>
          <span class="text-sm font-semibold">Regenerate questionnaire</span>
        </div>
      </div>
    </div>

    <div class="card mb-4" style="border:2px solid var(--secondary);background:linear-gradient(135deg,#f0faff 0%,#fff 100%);">
      <div class="card-header" style="background:transparent;">
        <h3 style="color:var(--secondary)">Quick Run — Built-in Systems</h3>
        <span class="badge badge-neutral text-xs">No external URLs required</span>
      </div>
      <div class="card-body">
        <p class="text-sm mb-4" style="color:var(--text-muted)">
          Run the evaluation using the platform's own Strict and Lenient RAG systems against your
          knowledge base. The system name and parameters above still apply — no external URLs needed.
        </p>
        <div class="flex items-center gap-4">
          <button class="btn btn-primary" id="btn-demo-strict" onclick="doDemoBuiltIn('strict')">
            ${icon('play')} Run Demo — Strict
          </button>
          <button class="btn btn-outline" id="btn-demo-lenient" onclick="doDemoBuiltIn('lenient')">
            ${icon('play')} Run Demo — Lenient
          </button>
        </div>
      </div>
    </div>

    <div id="eval-alert" class="hidden mb-4"></div>

    <div class="flex items-center gap-4">
      <button class="btn btn-primary btn-lg" id="btn-evaluate" onclick="doEvaluate()">
        ${icon('play')} Run Full Evaluation
      </button>
    </div>
    <div id="eval-job-area" class="hidden mt-6"></div>
  `;
}

function bindEvaluate() {}

async function doEvaluate() {
  const sysName    = document.getElementById('eval-sys-name').value.trim();
  const e2eUrl     = document.getElementById('eval-e2e-url').value.trim() || null;
  const retUrl     = document.getElementById('eval-ret-url').value.trim() || null;
  const genUrl     = document.getElementById('eval-gen-url').value.trim() || null;
  const alpha      = parseFloat(document.getElementById('alpha-slider').value);
  const regen      = document.getElementById('eval-regen').checked;
  const alertEl    = document.getElementById('eval-alert');

  if (!sysName) { showAlert(alertEl, 'error', 'Enter a system name.'); return; }

  alertEl.classList.add('hidden');
  const btn = document.getElementById('btn-evaluate');
  btn.disabled = true;

  try {
    const data = await apiPost('/api/evaluate', {
      rag_system_name: sysName,
      alpha,
      e2e_url:  e2eUrl,
      retriever_url: retUrl,
      generator_url: genUrl,
      regen_questionnaire: regen,
    });
    if (!data) { btn.disabled = false; return; }
    state.currentSystem = sysName;
    localStorage.setItem('rag_system', sysName);
    pollJob(data.job_id, 'eval-job-area', (result) => {
      showAlert(alertEl, 'success', 'Evaluation complete! View your results in the Metrics page.');
      btn.disabled = false;
    }, () => { btn.disabled = false; });
  } catch (e) {
    showAlert(alertEl, 'error', e.message);
    btn.disabled = false;
  }
}

async function doDemoBuiltIn(generator) {
  const sysName = (document.getElementById('eval-sys-name') || {}).value?.trim() || state.currentSystem;
  const alpha   = parseFloat((document.getElementById('alpha-slider') || {}).value || 0.8);
  const regen   = (document.getElementById('eval-regen') || {}).checked || false;
  const alertEl = document.getElementById('eval-alert');

  if (!sysName) { showAlert(alertEl, 'error', 'Enter a system name first.'); return; }

  alertEl.classList.add('hidden');
  const btn = document.getElementById(`btn-demo-${generator}`);
  if (btn) btn.disabled = true;
  const otherBtn = document.getElementById(`btn-demo-${generator === 'strict' ? 'lenient' : 'strict'}`);
  if (otherBtn) otherBtn.disabled = true;
  const mainBtn = document.getElementById('btn-evaluate');
  if (mainBtn) mainBtn.disabled = true;

  try {
    const data = await apiPost('/api/evaluate', {
      rag_system_name: sysName,
      alpha,
      regen_questionnaire: regen,
    });
    if (!data) {
      if (btn) btn.disabled = false;
      if (otherBtn) otherBtn.disabled = false;
      if (mainBtn) mainBtn.disabled = false;
      return;
    }
    state.currentSystem = sysName;
    localStorage.setItem('rag_system', sysName);

    const label = generator === 'strict' ? 'System A — Strict' : 'System B — Lenient';
    pollJob(data.job_id, 'eval-job-area', () => {
      if (btn) btn.disabled = false;
      if (otherBtn) otherBtn.disabled = false;
      if (mainBtn) mainBtn.disabled = false;
      showAlert(alertEl, 'success',
        `${label} evaluation complete! <a href="#" onclick="state.evalSystem='${generator}';navigate('results');return false;" style="font-weight:600;text-decoration:underline;">View Results →</a>`
      );
    }, () => {
      if (btn) btn.disabled = false;
      if (otherBtn) otherBtn.disabled = false;
      if (mainBtn) mainBtn.disabled = false;
    });
  } catch (e) {
    if (btn) btn.disabled = false;
    if (otherBtn) otherBtn.disabled = false;
    if (mainBtn) mainBtn.disabled = false;
    showAlert(alertEl, 'error', e.message);
  }
}

// ── Results page ───────────────────────────────────────────────────────────────
function renderResults() {
  const sys = state.currentSystem;
  return `
    <div class="page-header">
      <div class="section-label">Results</div>
      <h1 class="page-title">Evaluation Metrics</h1>
      <p class="page-desc">Performance breakdown across E2E, Retriever, and Generator evaluation modes.</p>
    </div>

    <div class="flex items-center gap-4 mb-4" style="flex-wrap:wrap;">
      <div class="form-group" style="min-width:200px;">
        <label class="form-label">System</label>
        <input class="form-input" id="results-sys" value="${escHtml(sys)}" placeholder="system name" />
      </div>
      <div class="form-group" style="min-width:180px;">
        <label class="form-label">Run</label>
        <select class="form-select" id="results-run">
          <option value="">Latest run</option>
        </select>
      </div>
      <div style="padding-top:1.4rem;">
        <button class="btn btn-primary" onclick="loadResults()">Load</button>
      </div>
    </div>

    <div class="system-tabs" id="system-tabs">
      <button class="system-tab active" data-sys="strict" onclick="switchSystemTab('strict')">System A — Strict</button>
      <button class="system-tab" data-sys="lenient" onclick="switchSystemTab('lenient')">System B — Lenient</button>
      <button class="system-tab" data-sys="external" onclick="switchSystemTab('external')">System C — External</button>
    </div>

    <div id="results-content">
      <div class="flex items-center gap-2 text-muted text-sm"><div class="spinner"></div> Loading metrics…</div>
    </div>
  `;
}

async function bindResults() {
  if (!state.currentSystem) return;
  // switchSystemTab handles both tab activation and data loading
  switchSystemTab(state.evalSystem || 'strict');
}

async function loadRunSelector() {
  const sysName = (document.getElementById('results-sys') || { value: state.currentSystem }).value.trim() || state.currentSystem;
  if (!sysName) return;
  try {
    const data = await apiFetch(`/api/runs?rag_system_name=${enc(sysName)}&system=${state.evalSystem}`);
    if (!data) return;
    const sel = document.getElementById('results-run');
    if (!sel) return;
    sel.innerHTML = '<option value="">Latest run</option>' +
      (data.runs || []).map(r => `<option value="${r.run_id}">${r.run_id} (${formatDate(r.created_at)})</option>`).join('');
  } catch (_) {}
}

async function loadResults() {
  const sysName = document.getElementById('results-sys')?.value.trim() || state.currentSystem;
  const runId   = document.getElementById('results-run')?.value || '';
  const content = document.getElementById('results-content');
  if (!sysName || !content) return;

  content.innerHTML = '<div class="flex items-center gap-2 text-muted text-sm"><div class="spinner"></div> Loading…</div>';

  try {
    const url = `/api/metrics?rag_system_name=${enc(sysName)}&system=${state.evalSystem}` + (runId ? `&run_id=${enc(runId)}` : '');
    const data = await apiFetch(url);
    if (!data || !data.metrics) {
      content.innerHTML = '<div class="alert alert-info">No metrics found. Run an evaluation first.</div>';
      return;
    }
    content.innerHTML = renderMetricsContent(data.metrics);
    initMetricCharts(data.metrics);
  } catch (e) {
    content.innerHTML = `<div class="alert alert-error">${e.message}</div>`;
  }
}

function switchSystemTab(sys) {
  state.evalSystem = sys;
  document.querySelectorAll('.system-tab').forEach(t => t.classList.toggle('active', t.dataset.sys === sys));
  loadRunSelector();
  loadResults();
}

function renderMetricsContent(m) {
  const e2e  = m.e2e  || {};
  const ret  = m.retriever  || {};
  const gen  = m.generator  || {};
  const attr = m.attribution || {};

  const pct = v => typeof v === 'number' ? (v * 100).toFixed(1) + '%' : '—';
  const num = v => typeof v === 'number' ? v.toFixed(3) : '—';
  const dlt = v => {
    if (typeof v !== 'number') return '—';
    const s = (v > 0 ? '+' : '') + (v * 100).toFixed(1) + '%';
    const cls = v > 0.02 ? 'delta-pos' : v < -0.02 ? 'delta-neg' : 'delta-zero';
    return `<span class="${cls}">${s}</span>`;
  };

  return `
    <!-- E2E metrics -->
    <div class="section-label mt-4">E2E Evaluation</div>
    <div class="metrics-grid">
      ${metricCard('E2E-OOKRI', num(e2e.E2E_OOKRI), 'Composite robustness score')}
      ${metricCard('OOKB Abstention', pct(e2e.E2E_AR), 'Out-of-KB refusal rate')}
      ${metricCard('OOKB Hallucination', pct(e2e.E2E_HR), 'Fabricated OOKB answers')}
      ${metricCard('IKB Coverage', pct(e2e.E2E_COV), 'Correctly answered in-KB queries')}
      ${metricCard('IKB Hallucination', pct(e2e.E2E_IK_HR), 'Wrong answers on in-KB queries')}
      ${metricCard('IKB Refusal Rate', pct(e2e.E2E_IK_RR), 'Safe misses on in-KB queries')}
    </div>

    <div class="chart-wrap mb-4">
      <canvas id="chart-e2e"></canvas>
    </div>

    <!-- Retriever -->
    <div class="section-label mt-4">Retriever</div>
    <div class="metrics-grid" style="grid-template-columns:1fr 1fr 1fr;">
      ${metricCard('E2E Recall@k', pct(e2e.R_Rec_k), 'IKB hit rate in E2E mode')}
      ${metricCard('Standalone Recall@k', pct(ret.R_Rec_k), 'Retriever-only hit rate')}
      ${metricCard('Standalone Precision@k', pct(ret.R_Prec_k), 'Retrieved chunk relevance')}
    </div>

    <!-- Generator -->
    <div class="section-label mt-4">Generator (Forced Context)</div>
    <div class="metrics-grid" style="grid-template-columns:1fr 1fr 1fr;">
      ${metricCard('G-AR', pct(gen.G_AR), 'Refuses on empty/irrelevant context')}
      ${metricCard('G-FAR', pct(gen.G_FAR), 'False abstention on correct context')}
      ${metricCard('G-PCCR', pct(gen.G_PCCR), 'Confabulation on partial context')}
    </div>

    <!-- Attribution delta -->
    <div class="section-label mt-4">Attribution Delta (E2E − Generator)</div>
    <div class="card">
      <div class="card-body">
        <p class="text-sm text-muted mb-4">Positive delta = retriever is causing the issue. Negative delta = retriever is helping.</p>
        <div class="delta-row">
          <span class="delta-label">Δ Hallucination</span>
          <span class="delta-value">${dlt(attr.delta_hallucination)}</span>
        </div>
        <div class="delta-row">
          <span class="delta-label">Δ Refusal</span>
          <span class="delta-value">${dlt(attr.delta_refusal)}</span>
        </div>
        <div class="delta-row">
          <span class="delta-label">Δ OOKRI</span>
          <span class="delta-value">${dlt(attr.delta_ookri)}</span>
        </div>
        <div class="chart-wrap mt-4" style="height:140px;">
          <canvas id="chart-delta"></canvas>
        </div>
      </div>
    </div>
  `;
}

function metricCard(name, value, desc) {
  return `
    <div class="metric-card">
      <div class="metric-name">${escHtml(name)}</div>
      <div class="metric-value">${value}</div>
      <div class="metric-desc">${escHtml(desc)}</div>
    </div>`;
}

function initMetricCharts(m) {
  destroyCharts();
  const e2e  = m.e2e  || {};
  const attr = m.attribution || {};

  // E2E bar chart
  const ctx1 = document.getElementById('chart-e2e');
  if (ctx1 && window.Chart) {
    state.chartInstances['e2e'] = new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: ['OOKB AR', 'OOKB HR', 'IKB Cov', 'IKB HR', 'IKB RR', 'Rec@k'],
        datasets: [{
          label: 'E2E Metrics',
          data: [e2e.E2E_AR, e2e.E2E_HR, e2e.E2E_COV, e2e.E2E_IK_HR, e2e.E2E_IK_RR, e2e.R_Rec_k].map(v => typeof v === 'number' ? +(v * 100).toFixed(1) : 0),
          backgroundColor: ['#10B981','#EF4444','#00A5E0','#F59E0B','#8B5CF6','#DA1884'],
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { min: 0, max: 100, ticks: { callback: v => v + '%' } } },
      },
    });
  }

  // Delta diverging bar
  const ctx2 = document.getElementById('chart-delta');
  if (ctx2 && window.Chart) {
    const dHall = typeof attr.delta_hallucination === 'number' ? +(attr.delta_hallucination * 100).toFixed(1) : 0;
    const dRef  = typeof attr.delta_refusal       === 'number' ? +(attr.delta_refusal * 100).toFixed(1) : 0;
    const dOok  = typeof attr.delta_ookri         === 'number' ? +(attr.delta_ookri * 100).toFixed(1) : 0;
    state.chartInstances['delta'] = new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: ['Δ Hallucination', 'Δ Refusal', 'Δ OOKRI'],
        datasets: [{
          data: [dHall, dRef, dOok],
          backgroundColor: [dHall, dRef, dOok].map(v => v > 0 ? 'rgba(239,68,68,.7)' : 'rgba(16,185,129,.7)'),
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { x: { ticks: { callback: v => v + 'pp' } } },
      },
    });
  }
}

function destroyCharts() {
  Object.values(state.chartInstances).forEach(c => { try { c.destroy(); } catch (_) {} });
  state.chartInstances = {};
}

// ── Insights page ──────────────────────────────────────────────────────────────
function renderInsights() {
  return `
    <div class="page-header">
      <div class="section-label">Results</div>
      <h1 class="page-title">LLM Diagnostic Insights</h1>
      <p class="page-desc">AI-generated root-cause analysis of your RAG evaluation results.</p>
    </div>

    <div class="flex items-center gap-4 mb-4" style="flex-wrap:wrap;">
      <div class="form-group" style="min-width:200px;">
        <label class="form-label">System</label>
        <input class="form-input" id="ins-sys" value="${escHtml(state.currentSystem)}" placeholder="system name" />
      </div>
      <div class="form-group" style="min-width:120px;">
        <label class="form-label">Variant</label>
        <select class="form-select" id="ins-variant">
          <option value="strict">System A — Strict</option>
          <option value="lenient">System B — Lenient</option>
          <option value="external">System C — External</option>
        </select>
      </div>
      <div style="padding-top:1.4rem;">
        <button class="btn btn-primary" id="btn-gen-insights" onclick="doGenInsights()">
          ${icon('lightbulb')} Generate Insights
        </button>
      </div>
    </div>

    <div id="insights-error" class="hidden mb-4"></div>
    <div id="insights-content"></div>
  `;
}

function bindInsights() {}

async function doGenInsights() {
  const sysName  = document.getElementById('ins-sys').value.trim() || state.currentSystem;
  const variant  = document.getElementById('ins-variant').value;
  const errorEl  = document.getElementById('ins-error') || document.getElementById('insights-error');
  const content  = document.getElementById('insights-content');
  const btn      = document.getElementById('btn-gen-insights');

  if (!sysName) { showAlert(errorEl, 'error', 'Enter a system name.'); return; }

  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Generating…';
  content.innerHTML = '';
  errorEl.classList.add('hidden');

  try {
    const mData = await apiFetch(`/api/metrics?rag_system_name=${enc(sysName)}&system=${variant}`);
    if (!mData || !mData.metrics) throw new Error('No metrics found — run evaluation first.');
    const alpha = mData.metrics.e2e?.alpha || 0.8;
    const ins   = await apiPost('/api/insights', { metrics: mData.metrics, alpha });
    if (!ins) return;
    content.innerHTML = renderInsightsReport(ins);
  } catch (e) {
    showAlert(errorEl, 'error', e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = icon('lightbulb') + ' Generate Insights';
  }
}

function renderInsightsReport(ins) {
  const bottleneckColor = {
    retrieval:  'badge-warning',
    generation: 'badge-danger',
    both:       'badge-danger',
    none:       'badge-success',
  }[ins.primary_bottleneck] || 'badge-neutral';

  const safetyColor = { safe: 'badge-success', at_risk: 'badge-warning', unsafe: 'badge-danger' }[ins.safety_status] || 'badge-neutral';
  const helpColor   = { helpful: 'badge-success', 'over-refusing': 'badge-warning', unreliable: 'badge-danger' }[ins.helpfulness_status] || 'badge-neutral';

  const recs = (ins.recommendations || []).map(r => `<li>${escHtml(r)}</li>`).join('');

  return `
    <div class="flex items-center gap-3 mb-4" style="flex-wrap:wrap;">
      <span class="badge ${bottleneckColor}">Bottleneck: ${escHtml(ins.primary_bottleneck)}</span>
      <span class="badge ${safetyColor}">Safety: ${escHtml(ins.safety_status)}</span>
      <span class="badge ${helpColor}">Helpfulness: ${escHtml(ins.helpfulness_status)}</span>
    </div>

    <div class="insight-card">
      <div class="insight-card-header"><h3>Retrieval Diagnosis</h3></div>
      <div class="insight-card-body">${escHtml(ins.retrieval_diagnosis || '—')}</div>
    </div>

    <div class="insight-card">
      <div class="insight-card-header"><h3>Generation Diagnosis</h3></div>
      <div class="insight-card-body">${escHtml(ins.generation_diagnosis || '—')}</div>
    </div>

    <div class="insight-card">
      <div class="insight-card-header"><h3>Recommendations</h3></div>
      <div class="insight-card-body"><ul class="rec-list">${recs}</ul></div>
    </div>

    <div class="insight-card">
      <div class="insight-card-header">
        <h3>Full Narrative</h3>
        <button class="narrative-toggle" onclick="toggleNarrative(this)">Show ▼</button>
      </div>
      <div class="insight-card-body">
        <div class="narrative-body" id="narrative-body">${escHtml(ins.full_narrative || '—').replace(/\n/g, '<br>')}</div>
      </div>
    </div>
  `;
}

function toggleNarrative(btn) {
  const body = document.getElementById('narrative-body');
  if (!body) return;
  const open = body.classList.toggle('open');
  btn.textContent = open ? 'Hide ▲' : 'Show ▼';
}

// ── Demo page ──────────────────────────────────────────────────────────────────
function renderDemo() {
  return `
    <div class="page-header">
      <div class="section-label">Tools</div>
      <h1 class="page-title">Demo RAG</h1>
      <p class="page-desc">Query the built-in strict and lenient RAG systems interactively.</p>
    </div>

    <div class="demo-tabs">
      <button class="demo-tab active" id="dtab-e2e" onclick="switchDemoTab('e2e')">E2E Query</button>
      <button class="demo-tab" id="dtab-retriever" onclick="switchDemoTab('retriever')">Retriever</button>
      <button class="demo-tab" id="dtab-generator" onclick="switchDemoTab('generator')">Generator</button>
    </div>

    <div id="demo-e2e">
      <div class="flex items-center gap-4 mb-4" style="flex-wrap:wrap;">
        <div class="form-group flex-1" style="min-width:250px;">
          <label class="form-label">System Name</label>
          <input class="form-input" id="demo-sys" value="${escHtml(state.currentSystem)}" placeholder="my-rag" />
        </div>
        <div class="form-group">
          <label class="form-label">Generator</label>
          <select class="form-select" id="demo-gen">
            <option value="strict">Strict (System A)</option>
            <option value="lenient">Lenient (System B)</option>
          </select>
        </div>
      </div>
      <div class="form-group mb-3">
        <label class="form-label">Question</label>
        <div class="flex gap-2">
          <input class="form-input flex-1" id="demo-q" type="text" placeholder="Ask anything about your knowledge base…" />
          <button class="btn btn-primary" id="btn-demo-e2e" onclick="doDemoE2E()">Ask</button>
        </div>
      </div>
      <div id="demo-e2e-result" class="hidden">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-semibold">Response</h3>
          <span class="latency-badge" id="demo-e2e-latency"></span>
        </div>
        <div class="response-box" id="demo-e2e-response"></div>
        <div class="chunks-accordion mt-3">
          <button class="chunk-toggle" onclick="toggleChunks('demo-e2e-chunks')">
            ▶ Retrieved chunks
          </button>
          <div class="chunk-list hidden" id="demo-e2e-chunks"></div>
        </div>
      </div>
    </div>

    <div id="demo-retriever" class="hidden">
      <div class="flex items-center gap-4 mb-4">
        <div class="form-group flex-1">
          <label class="form-label">System Name</label>
          <input class="form-input" id="demo-ret-sys" value="${escHtml(state.currentSystem)}" placeholder="my-rag" />
        </div>
      </div>
      <div class="form-group mb-3">
        <label class="form-label">Query</label>
        <div class="flex gap-2">
          <input class="form-input flex-1" id="demo-ret-q" type="text" placeholder="Retrieve relevant chunks…" />
          <button class="btn btn-primary" id="btn-demo-ret" onclick="doDemoRetrieve()">Retrieve</button>
        </div>
      </div>
      <div class="chunk-list" id="demo-ret-results"></div>
    </div>

    <div id="demo-generator" class="hidden">
      <div class="flex items-center gap-4 mb-4" style="flex-wrap:wrap;">
        <div class="form-group">
          <label class="form-label">Generator</label>
          <select class="form-select" id="demo-gen-type">
            <option value="strict">Strict (System A)</option>
            <option value="lenient">Lenient (System B)</option>
          </select>
        </div>
      </div>
      <div class="form-group mb-3">
        <label class="form-label">Question</label>
        <input class="form-input" id="demo-gen-q" type="text" placeholder="Question to answer with forced context…" />
      </div>
      <div class="form-group mb-3">
        <label class="form-label">Forced Context</label>
        <textarea class="form-textarea" id="demo-gen-ctx" placeholder="Paste context passages here (one per line)…" rows="5"></textarea>
      </div>
      <button class="btn btn-primary" id="btn-demo-gen" onclick="doDemoGenerate()">Generate</button>
      <div id="demo-gen-result" class="hidden mt-3">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-semibold">Response</h3>
          <span class="latency-badge" id="demo-gen-latency"></span>
        </div>
        <div class="response-box" id="demo-gen-response"></div>
      </div>
    </div>
  `;
}

function bindDemo() {
  ['demo-q', 'demo-ret-q', 'demo-gen-q'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('keydown', e => { if (e.key === 'Enter') { } });
  });
  document.getElementById('demo-q')?.addEventListener('keydown', e => { if (e.key === 'Enter') doDemoE2E(); });
  document.getElementById('demo-ret-q')?.addEventListener('keydown', e => { if (e.key === 'Enter') doDemoRetrieve(); });
}

function switchDemoTab(tab) {
  ['e2e', 'retriever', 'generator'].forEach(t => {
    document.getElementById(`demo-${t}`)?.classList.toggle('hidden', t !== tab);
    document.getElementById(`dtab-${t}`)?.classList.toggle('active', t === tab);
  });
}

async function doDemoE2E() {
  const sys = document.getElementById('demo-sys').value.trim();
  const q   = document.getElementById('demo-q').value.trim();
  const gen = document.getElementById('demo-gen').value;
  const btn = document.getElementById('btn-demo-e2e');
  if (!sys || !q) return;
  btn.disabled = true; btn.textContent = '…';
  const t0 = Date.now();
  try {
    const data = await apiPost('/api/demo/query', { question: q, rag_system_name: sys, generator: gen });
    if (!data) return;
    const res = document.getElementById('demo-e2e-result');
    res.classList.remove('hidden');
    document.getElementById('demo-e2e-response').textContent = data.response;
    document.getElementById('demo-e2e-latency').textContent = `${Date.now() - t0} ms`;
    const chunksList = document.getElementById('demo-e2e-chunks');
    chunksList.innerHTML = (data.retrieved_ids || []).map((id, i) =>
      `<div class="chunk-item"><div class="chunk-id">${escHtml(id)}</div>${escHtml((data.retrieved_contexts || [])[i] || '')}</div>`
    ).join('');
  } catch (e) {
    document.getElementById('demo-e2e-result').classList.remove('hidden');
    document.getElementById('demo-e2e-response').textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = 'Ask';
  }
}

async function doDemoRetrieve() {
  const sys = document.getElementById('demo-ret-sys').value.trim();
  const q   = document.getElementById('demo-ret-q').value.trim();
  const btn = document.getElementById('btn-demo-ret');
  if (!sys || !q) return;
  btn.disabled = true; btn.textContent = '…';
  try {
    const data = await apiPost('/api/demo/retrieve', { question: q, rag_system_name: sys });
    if (!data) return;
    document.getElementById('demo-ret-results').innerHTML = (data.chunks || []).map(c =>
      `<div class="chunk-item"><div class="chunk-id">${escHtml(c.id)} <span class="text-muted">(score: ${(c.score || 0).toFixed(3)})</span></div>${escHtml(c.content)}</div>`
    ).join('');
  } catch (e) {
    document.getElementById('demo-ret-results').innerHTML = `<div class="alert alert-error">${e.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Retrieve';
  }
}

async function doDemoGenerate() {
  const q    = document.getElementById('demo-gen-q').value.trim();
  const ctx  = document.getElementById('demo-gen-ctx').value.trim().split('\n').filter(Boolean);
  const gen  = document.getElementById('demo-gen-type').value;
  const btn  = document.getElementById('btn-demo-gen');
  if (!q) return;
  btn.disabled = true; btn.textContent = '…';
  const t0 = Date.now();
  try {
    const data = await apiPost('/api/demo/generate', { question: q, context: ctx, generator: gen });
    if (!data) return;
    document.getElementById('demo-gen-result').classList.remove('hidden');
    document.getElementById('demo-gen-response').textContent = data.response;
    document.getElementById('demo-gen-latency').textContent = `${Date.now() - t0} ms`;
  } catch (e) {
    document.getElementById('demo-gen-result').classList.remove('hidden');
    document.getElementById('demo-gen-response').textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = 'Generate';
  }
}

// ── Job polling ────────────────────────────────────────────────────────────────
function pollJob(jobId, containerId, onDone, onFail) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.classList.remove('hidden');
  container.innerHTML = renderJobCard(jobId, { status: 'pending', step: 'queued', progress: 0, message: 'Job queued' });

  const interval = setInterval(async () => {
    try {
      const job = await apiFetch(`/api/jobs/${jobId}`);
      if (!job) { clearInterval(interval); return; }
      container.innerHTML = renderJobCard(jobId, job);
      if (job.status === 'done') {
        clearInterval(interval);
        if (onDone) onDone(job.result);
      } else if (job.status === 'failed') {
        clearInterval(interval);
        if (onFail) onFail(job.error);
      }
    } catch (_) {}
  }, 5000);
}

function renderJobCard(jobId, job) {
  const isDone   = job.status === 'done';
  const isFailed = job.status === 'failed';
  const color    = isDone ? 'var(--success)' : isFailed ? 'var(--danger)' : 'var(--secondary)';

  return `
    <div class="card">
      <div class="card-header">
        <h3 style="color:${color}">${isDone ? '✓ Complete' : isFailed ? '✗ Failed' : '⟳ Running'}</h3>
        <span class="badge ${isDone ? 'badge-success' : isFailed ? 'badge-danger' : 'badge-secondary'}">${escHtml(job.status)}</span>
      </div>
      <div class="card-body">
        <div class="flex items-center gap-3 mb-3">
          <div class="flex-1">
            <div class="text-sm font-semibold mb-1">${escHtml(job.message || '')}</div>
            <div class="progress-wrap"><div class="progress-bar" style="width:${job.progress || 0}%"></div></div>
          </div>
          <span class="text-sm text-muted">${job.progress || 0}%</span>
        </div>
        ${isFailed ? `<div class="alert alert-error">${escHtml(job.error || 'Unknown error')}</div>` : ''}
        ${isDone && job.result ? `<div class="alert alert-success">Job completed successfully.</div>` : ''}
      </div>
    </div>
  `;
}

// ── Utility helpers ────────────────────────────────────────────────────────────
function escHtml(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function enc(s) { return encodeURIComponent(s); }

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function formatDate(iso) {
  if (!iso) return '';
  try { return new Date(iso).toLocaleString(); } catch (_) { return iso; }
}

function showAlert(el, type, msg) {
  if (!el) return;
  el.className = `alert alert-${type} mb-4`;
  el.textContent = msg;
  el.classList.remove('hidden');
}

function toggleChunks(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle('hidden');
  const btn = el.previousElementSibling;
  if (btn) btn.textContent = el.classList.contains('hidden') ? '▶ Retrieved chunks' : '▼ Retrieved chunks';
}

// ── SVG icons ──────────────────────────────────────────────────────────────────
function icon(name) {
  const icons = {
    'grid':       '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>',
    'upload':     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>',
    'play':       '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>',
    'bar-chart':  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    'lightbulb':  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="9" y1="18" x2="15" y2="18"/><line x1="10" y1="22" x2="14" y2="22"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/></svg>',
    'message':    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    'list':       '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>',
  };
  return icons[name] || '';
}

// ── Shell binder ───────────────────────────────────────────────────────────────
function bindShell() {}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', renderApp);
