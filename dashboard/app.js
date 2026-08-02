/* FLEETWATCH dashboard client
   Connects to /ws/live for push updates; falls back to polling
   /api/fleet/units if the socket drops. Renders per-unit gauge cards
   and a live trend chart for the selected unit. */

const API_BASE = "";
let selectedUnit = null;
let unitState = {};       // unit_id -> latest prediction
let unitHistory = {};     // unit_id -> array of recent predictions (client-side cache)
let trendChart = null;
let socket = null;
let pollTimer = null;

const MAX_POINTS = 40;

function setConnStatus(state){
  const el = document.getElementById("connStatus");
  const label = document.getElementById("connLabel");
  el.classList.remove("live", "down");
  if(state === "live"){ el.classList.add("live"); label.textContent = "LIVE"; }
  else if(state === "down"){ el.classList.add("down"); label.textContent = "RECONNECTING"; }
  else { label.textContent = "CONNECTING"; }
}

function connectWebSocket(){
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws/live`;
  socket = new WebSocket(url);

  socket.onopen = () => { setConnStatus("live"); stopPolling(); };
  socket.onclose = () => { setConnStatus("down"); startPolling(); setTimeout(connectWebSocket, 3000); };
  socket.onerror = () => { socket.close(); };

  socket.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if(msg.type === "snapshot" || msg.type === "tick"){
      msg.predictions.forEach(applyPrediction);
      renderAll();
    }
  };
}

function startPolling(){
  if(pollTimer) return;
  pollTimer = setInterval(async () => {
    try{
      const res = await fetch(`${API_BASE}/api/fleet/units`);
      const data = await res.json();
      data.units.forEach(applyPrediction);
      renderAll();
    }catch(e){ /* server likely still starting */ }
  }, 2000);
}
function stopPolling(){ if(pollTimer){ clearInterval(pollTimer); pollTimer = null; } }

function applyPrediction(p){
  if(p.error) return;
  unitState[p.unit_id] = p;
  if(!unitHistory[p.unit_id]) unitHistory[p.unit_id] = [];
  const h = unitHistory[p.unit_id];
  h.push(p);
  if(h.length > MAX_POINTS) h.shift();
}

function renderAll(){
  renderKpis();
  renderUnitGrid();
  if(selectedUnit && unitState[selectedUnit]) renderDetail(selectedUnit);
}

function renderKpis(){
  const units = Object.values(unitState);
  if(units.length === 0) return;
  const healthy = units.filter(u => u.status === "healthy").length;
  const warning = units.filter(u => u.status === "warning").length;
  const critical = units.filter(u => u.status === "critical").length;
  const avgRisk = units.reduce((s,u) => s + u.risk_score, 0) / units.length;

  document.getElementById("kpiTotal").textContent = units.length;
  document.getElementById("kpiHealthy").textContent = healthy;
  document.getElementById("kpiWarning").textContent = warning;
  document.getElementById("kpiCritical").textContent = critical;
  document.getElementById("kpiRisk").textContent = avgRisk.toFixed(3);
}

function gaugeSvg(riskScore, statusClass){
  // semicircular gauge, 0..1 mapped to -90deg..+90deg needle rotation
  const angle = -90 + Math.min(Math.max(riskScore, 0), 1) * 180;
  const color = statusClass === "critical" ? "var(--crit)" : statusClass === "warning" ? "var(--warn)" : "var(--ok)";
  return `
  <svg class="gauge-svg" viewBox="0 0 78 52">
    <path d="M6,46 A33,33 0 0,1 72,46" fill="none" stroke="#2c3542" stroke-width="6" stroke-linecap="round"/>
    <path d="M6,46 A33,33 0 0,1 72,46" fill="none" stroke="${color}" stroke-width="6" stroke-linecap="round"
          stroke-dasharray="${Math.min(Math.max(riskScore,0),1) * 103.6} 200"/>
    <line class="gauge-needle" x1="39" y1="46" x2="39" y2="16" stroke="#e8ecf1" stroke-width="2"
          transform="rotate(${angle} 39 46)"/>
    <circle cx="39" cy="46" r="3.5" fill="#e8ecf1"/>
  </svg>`;
}

function renderUnitGrid(){
  const grid = document.getElementById("unitGrid");
  const ids = Object.keys(unitState).sort();
  if(ids.length === 0){
    grid.innerHTML = `<div class="panel-note">Waiting for telemetry…</div>`;
    return;
  }
  if(!selectedUnit) selectedUnit = ids[0];

  grid.innerHTML = ids.map(id => {
    const p = unitState[id];
    const sel = id === selectedUnit ? "selected" : "";
    return `
    <div class="unit-card ${sel}" data-unit="${id}">
      <div class="unit-card-head">
        <div>
          <div class="unit-id">${id}</div>
          <div class="unit-type">${p.equipment_type || ""}</div>
        </div>
        <span class="status-pill ${p.status}">${p.status}</span>
      </div>
      <div class="gauge-row">
        ${gaugeSvg(p.risk_score, p.status)}
        <div class="unit-metrics">
          <span>Risk <b>${p.risk_score.toFixed(3)}</b></span>
          <span>P(fail) <b>${(p.failure_probability*100).toFixed(2)}%</b></span>
          <span class="unit-cycle">cycle ${p.cycle}</span>
        </div>
      </div>
    </div>`;
  }).join("");

  grid.querySelectorAll(".unit-card").forEach(card => {
    card.addEventListener("click", () => {
      selectedUnit = card.dataset.unit;
      renderUnitGrid();
      renderDetail(selectedUnit);
    });
  });
}

function renderDetail(unitId){
  const p = unitState[unitId];
  if(!p) return;
  document.getElementById("detailUnitLabel").textContent =
    `${unitId} · ${p.equipment_type} · cycle ${p.cycle}`;
  document.getElementById("detailFailureProb").textContent = (p.failure_probability*100).toFixed(2) + "%";
  document.getElementById("detailAnomalyScore").textContent = p.anomaly_score.toFixed(4);
  document.getElementById("detailCycle").textContent = p.cycle;
  const statusEl = document.getElementById("detailStatus");
  statusEl.textContent = p.status.toUpperCase();
  statusEl.style.color = p.status === "critical" ? "var(--crit)" : p.status === "warning" ? "var(--warn)" : "var(--ok)";

  const hist = unitHistory[unitId] || [];
  const labels = hist.map(h => h.cycle);
  const riskData = hist.map(h => h.risk_score);
  const failData = hist.map(h => h.failure_probability);

  if(!trendChart){
    const ctx = document.getElementById("trendChart").getContext("2d");
    trendChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Risk score", data: riskData, borderColor: "#4da3ff", backgroundColor: "rgba(77,163,255,0.08)",
            fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
          { label: "Failure probability", data: failData, borderColor: "#f04848", backgroundColor: "transparent",
            fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2, borderDash: [4,3] },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { ticks: { color: "#5d6b7c", font: { family: "IBM Plex Mono", size: 10 } }, grid: { color: "#212831" } },
          y: { min: 0, max: 1, ticks: { color: "#5d6b7c", font: { family: "IBM Plex Mono", size: 10 } }, grid: { color: "#212831" } },
        },
        plugins: { legend: { labels: { color: "#9aa7b6", font: { family: "IBM Plex Sans", size: 11 } } } }
      }
    });
  } else {
    trendChart.data.labels = labels;
    trendChart.data.datasets[0].data = riskData;
    trendChart.data.datasets[1].data = failData;
    trendChart.update();
  }
}

async function loadModelMetrics(){
  try{
    const res = await fetch(`${API_BASE}/api/metrics`);
    if(!res.ok) return;
    const data = await res.json();
    const grid = document.getElementById("metricsGrid");
    let html = "";
    if(data.classifier_metrics){
      html += `<div class="metric-section-title">XGBoost failure classifier</div>`;
      html += metricBlocks(data.classifier_metrics, ["accuracy","precision","recall","f1","roc_auc","pr_auc","mcc","brier_score"]);
      document.getElementById("kpiAuc").textContent = data.classifier_metrics.roc_auc.toFixed(3);
    }
    if(data.autoencoder_metrics){
      html += `<div class="metric-section-title">Autoencoder anomaly detector</div>`;
      html += metricBlocks(data.autoencoder_metrics, ["accuracy","precision","recall","f1","roc_auc","pr_auc","mean_reconstruction_error"]);
    }
    grid.innerHTML = html;
  }catch(e){ /* metrics not trained yet */ }
}

async function loadDatasetOptions(){
  try {
    const res = await fetch(`${API_BASE}/api/dataset/available`);
    if(!res.ok){ console.error("dataset/available failed:", res.status); return; }
    const data = await res.json();
    console.log("available subsets:", data.available_subsets);
    const sel = document.getElementById("datasetSelect");
    if(!sel){ console.error("datasetSelect element not found in DOM"); return; }
    if(data.available_subsets.length === 0){
      sel.innerHTML = `<option>no trained models found</option>`;
      return;
    }
    sel.innerHTML = data.available_subsets.map(s =>
      `<option value="${s}" ${s === data.current_subset ? "selected" : ""}>${s}</option>`).join("");
  } catch(e){ console.error("loadDatasetOptions error:", e); }
}

async function onDatasetChange(e){
  const res = await fetch(`${API_BASE}/api/dataset/select?subset=${e.target.value}`, { method: "POST" });
  if(!res.ok){ console.error("dataset/select failed:", res.status, await res.text()); return; }
  unitState = {}; unitHistory = {}; selectedUnit = null;
  if(trendChart){ trendChart.destroy(); trendChart = null; }
  renderAll();
}

document.getElementById("datasetSelect")?.addEventListener("change", onDatasetChange);



function metricBlocks(metrics, keys){
  return keys.filter(k => k in metrics).map(k => `
    <div class="metric-block">
      <div class="m-label">${k.replace(/_/g," ")}</div>
      <div class="m-value">${typeof metrics[k] === "number" ? metrics[k].toFixed(4) : metrics[k]}</div>
    </div>`).join("");
}

setConnStatus("connecting");
connectWebSocket();
loadModelMetrics();
setInterval(loadModelMetrics, 60000);
loadDatasetOptions();