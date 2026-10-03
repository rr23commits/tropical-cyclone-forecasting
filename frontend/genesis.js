const $ = id => document.getElementById(id);
const stamp = value => new Date(value).toLocaleString("en-GB", {timeZone:"UTC", month:"short", day:"2-digit", year:"numeric", hour:"2-digit", minute:"2-digit", hour12:false}).replace(",", "") + " UTC";
let data, selected;

fetch("data/genesis.json").then(response => response.ok ? response.text() : Promise.reject(Error("Missing Genesis imagery export.")))
  // The export uses NaN for an unavailable basin; normalize it only in memory.
  .then(text => JSON.parse(text.replace(/\bNaN\b/g, "null"))).then(init)
  .catch(error => { $("genesisApp").innerHTML = `<div class="error">${error.message}</div>`; });

function init(payload) {
  data = payload;
  $("genesisApp").innerHTML = `
    <header class="labbar"><a class="brand" href="index.html"><span class="brand-mark">◉</span> TROP-CYC <span>/</span> RESEARCH LAB</a><span class="lab-divider"></span><span class="page-tag">GENESIS IMAGERY</span><nav class="page-nav" aria-label="Research views"><a href="index.html">Spatial Trajectory</a><a href="monthly.html">Monthly Activity</a></nav></header>
    <section class="intro"><div><p class="eyebrow">GRID SATELLITE IR / CNN-GRU</p><h1>Genesis imagery</h1><p class="lede">An <b>exploratory imagery experiment</b> using nine 201×201 GridSat IR frames at three-hour spacing to classify the frozen 24-hour Genesis label. This static viewer shows archived inputs only; it performs no browser inference.</p></div><div class="intro-meta"><span>9 IR FRAMES</span><span>3-HOUR SPACING</span><span>EXPLORATORY</span></div></section>
    <section class="panel viewer"><div class="panel-head"><div><p class="eyebrow">COMPLETE CANDIDATE HISTORY</p><h2>GridSat IR sequence</h2></div><label class="candidate-control">SELECT CANDIDATE<select id="candidateSelect"></select></label></div><div class="candidate-meta" id="candidateMeta"></div><div id="frames" class="frames" aria-live="polite"></div><p class="chart-note">Chronological context window, from t−24h through issue time t. Images are archived 256×256 static renderings of the experiment inputs.</p></section>
    <section class="metrics-strip"><div><p class="eyebrow">EXPERIMENT STATUS</p><h2>Exploratory CNN-GRU</h2></div><div class="metric"><b>ROC-AUC</b><span id="rocAuc"></span></div><div class="metric"><b>F1</b><span id="f1"></span></div><div class="metric-copy">Held-out test metrics. This imagery classifier is separate from the main Spatial Trajectory forecasting model.</div></section>
    <footer><span>STATIC RESEARCH PRESENTATION / NON-OPERATIONAL</span><span id="sourceCount"></span></footer>`;
  const select = $("candidateSelect");
  data.candidates.forEach((candidate, index) => select.add(new Option(`TCC-${candidate.candidate_id} · ${candidate.split.toUpperCase()} · ${candidate.issue_time_utc}`, index)));
  select.onchange = event => { selected = data.candidates[+event.target.value]; render(); };
  selected = data.candidates[0];
  $("rocAuc").textContent = data.experiment.test_roc_auc.toFixed(3);
  $("f1").textContent = data.experiment.test_f1.toFixed(3);
  $("sourceCount").textContent = `${data.candidate_count.toLocaleString()} COMPLETE CANDIDATES // ${data.image_count.toLocaleString()} ARCHIVED FRAMES`;
  render();
}

function render() {
  const basin = selected.basin || "UNRESOLVED BASIN";
  const position = `${Math.abs(selected.current_lat).toFixed(1)}°${selected.current_lat >= 0 ? "N" : "S"} · ${Math.abs(selected.current_lon).toFixed(1)}°${selected.current_lon >= 0 ? "E" : "W"}`;
  $("candidateMeta").innerHTML = `<div><small>CANDIDATE / SPLIT</small><b>TCC-${selected.candidate_id} · ${selected.split.toUpperCase()}</b></div><div><small>ISSUE TIME</small><b>${stamp(selected.issue_time_utc)}</b></div><div><small>LOCATION / BASIN</small><b>${position} · ${basin}</b></div><div><small>LABEL_24H</small><b class="label ${selected.label_24h ? "positive" : "negative"}">${selected.label_24h ? "1 · GENESIS" : "0 · NO GENESIS"}</b></div><div><small>CNN-GRU PROBABILITY</small><b class="unavailable">Not exported</b></div>`;
  $("frames").innerHTML = selected.frames.map(frame => `<figure><div class="frame-head"><b>${frame.offset}</b><span>${stamp(frame.timestamp_utc)}</span></div><img src="${frame.image}" alt="GridSat IR image for TCC-${selected.candidate_id} at ${frame.timestamp_utc}" loading="lazy" /><figcaption>IR brightness temperature</figcaption></figure>`).join("");
}
