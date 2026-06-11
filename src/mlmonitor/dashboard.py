"""Single-file HTML dashboard served at GET /dashboard.

Pure stdlib-on-the-server, vanilla JS in the browser — polls the existing
/monitor/checks and /monitor/reports endpoints, no build step.
"""

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Agentic MLOps Monitor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#0f1419; --card:#1a2129; --text:#e6e8ea; --muted:#8b97a3;
          --ok:#2ecc71; --warn:#f1c40f; --alert:#e74c3c; --accent:#4aa3ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 system-ui, "Segoe UI", sans-serif; padding:24px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); margin-bottom:20px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
          gap:12px; margin-bottom:20px; }
  .card { background:var(--card); border-radius:10px; padding:14px 16px; }
  .card .label { color:var(--muted); font-size:12px; text-transform:uppercase;
                 letter-spacing:.06em; }
  .card .value { font-size:26px; font-weight:600; margin-top:4px; }
  .status-ok { color:var(--ok); } .status-warn { color:var(--warn); }
  .status-alert { color:var(--alert); } .status-no_data, .status-insufficient_data { color:var(--muted); }
  table { width:100%; border-collapse:collapse; background:var(--card);
          border-radius:10px; overflow:hidden; }
  th, td { padding:9px 12px; text-align:left; }
  th { color:var(--muted); font-size:12px; text-transform:uppercase;
       letter-spacing:.05em; border-bottom:1px solid #2a3340; }
  tr:not(:last-child) td { border-bottom:1px solid #222b35; }
  .badge { padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
  .b-ok { background:#163b2a; color:var(--ok); } .b-warn { background:#3d3614; color:var(--warn); }
  .b-alert { background:#3d1a16; color:var(--alert); }
  .b-no_data, .b-insufficient_data { background:#262e37; color:var(--muted); }
  h2 { font-size:15px; margin:24px 0 10px; color:var(--muted); }
  .diag { max-width:520px; white-space:normal; color:var(--muted); }
  .spark { display:flex; align-items:flex-end; gap:2px; height:36px; }
  .spark div { width:8px; background:var(--accent); border-radius:2px 2px 0 0; min-height:2px; }
  .footer { color:var(--muted); font-size:12px; margin-top:18px; }
</style>
</head>
<body>
<h1>Agentic MLOps Monitor</h1>
<div class="sub">Live drift checks &amp; agent verdicts · auto-refreshes every 15s</div>

<div class="grid">
  <div class="card"><div class="label">Latest status</div><div class="value" id="kpi-status">–</div></div>
  <div class="card"><div class="label">PSI max</div><div class="value" id="kpi-psi">–</div></div>
  <div class="card"><div class="label">Production F1</div><div class="value" id="kpi-f1">–</div></div>
  <div class="card"><div class="label">F1 drop</div><div class="value" id="kpi-drop">–</div></div>
  <div class="card"><div class="label">PSI trend (recent)</div><div class="spark" id="spark"></div></div>
</div>

<h2>Recent drift checks</h2>
<table>
  <thead><tr><th>#</th><th>Time (UTC)</th><th>Status</th><th>PSI max</th><th>PSI mean</th><th>F1</th><th>F1 drop</th></tr></thead>
  <tbody id="checks"></tbody>
</table>

<h2>Agent reports</h2>
<table>
  <thead><tr><th>#</th><th>Time (UTC)</th><th>Diagnosis</th><th>Retrain?</th></tr></thead>
  <tbody id="reports"></tbody>
</table>

<div class="footer">Endpoints: <code>/docs</code> · <code>/metrics</code> (Prometheus) · <code>/monitor/checks</code></div>

<script>
const fmt = (v, d=3) => (v === null || v === undefined) ? "–" : Number(v).toFixed(d);
const badge = s => `<span class="badge b-${s}">${s}</span>`;

async function refresh() {
  try {
    const [checks, reports] = await Promise.all([
      fetch("/monitor/checks?limit=15").then(r => r.json()),
      fetch("/monitor/reports?limit=10").then(r => r.json()),
    ]);
    if (checks.length) {
      const c = checks[0];
      const el = document.getElementById("kpi-status");
      el.textContent = c.status; el.className = "value status-" + c.status;
      document.getElementById("kpi-psi").textContent = fmt(c.psi_max);
      document.getElementById("kpi-f1").textContent = fmt(c.perf_f1);
      document.getElementById("kpi-drop").textContent = fmt(c.perf_drop);
      const maxPsi = Math.max(...checks.map(x => x.psi_max), 0.001);
      document.getElementById("spark").innerHTML = checks.slice().reverse()
        .map(x => `<div style="height:${Math.max(2, 100 * x.psi_max / maxPsi * 0.36)}px"
                        title="${fmt(x.psi_max)}"></div>`).join("");
    }
    document.getElementById("checks").innerHTML = checks.map(c => `
      <tr><td>${c.id}</td><td>${c.ts.replace("T"," ").slice(0,19)}</td>
      <td>${badge(c.status)}</td><td>${fmt(c.psi_max)}</td><td>${fmt(c.psi_mean)}</td>
      <td>${fmt(c.perf_f1)}</td><td>${fmt(c.perf_drop)}</td></tr>`).join("");
    document.getElementById("reports").innerHTML = reports.map(r => `
      <tr><td>${r.id}</td><td>${r.ts.replace("T"," ").slice(0,19)}</td>
      <td class="diag">${r.diagnosis}</td>
      <td>${r.triggered_retraining ? "🔁 dispatched" : "—"}</td></tr>`).join("");
  } catch (e) { console.error(e); }
}
refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>"""
