(() => {
  "use strict";
  const DATA = JSON.parse(document.getElementById("vmax-data").textContent);
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const m = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : (v >= 0 ? "+" : "") + v.toFixed(d));
  const abs = (v, d = 3) => (v === null || v === undefined ? "—" : v.toFixed(d));
  const pct = (v, d = 1) => (v === null || v === undefined ? "—" : (v * 100).toFixed(d) + "%");
  const pretty = (s) => s.replace(/_/g, " ").replace("violation ", "excursion ");
  const SVGNS = "http://www.w3.org/2000/svg";

  function el(tag, attrs = {}, parent = null) {
    const node = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  /* ---------------------------------------------------------------- header */
  const meta = DATA.meta;
  const anyCase = DATA.cases[0] || {};
  $("meta").innerHTML = [
    ["detector", anyCase.detector || "vmaxnet"],
    ["held-out camera", meta.held_out_camera],
    ["clips judged", meta.clip_count],
    ["generated", meta.generated_utc],
  ].map(([k, v]) => `<span class="chip">${esc(k)} <b>${esc(v)}</b></span>`).join("");

  /* ------------------------------------------------------------ headline */
  const heldCam = DATA.summary["held_out_camera"] || {};
  const heldScen = DATA.summary["held_out_scenario"] || {};
  const surveyed = DATA.summary["mode:surveyed"] || {};
  const gradErr = DATA.graduated.filter((g) => g.true_peak_m !== null && g.true_peak_m !== undefined)
    .map((g) => Math.abs(g.detected_peak_m - g.true_peak_m));
  const medianPeak = gradErr.length ? gradErr.slice().sort((a, b) => a - b)[Math.floor(gradErr.length / 2)] : null;
  const attributed = DATA.cases.reduce((a, c) => a + (c.attribution.attributed || 0), 0);
  const attrCorrect = DATA.cases.reduce((a, c) => a + (c.attribution.correct || 0), 0);

  $("tiles").innerHTML = [
    {
      v: pct(heldCam.recall ?? surveyed.recall, 1), k: "Car recall, unseen camera",
      n: `${heldCam.clips || 0} clips shot from the camera never trained on`,
    },
    {
      v: (surveyed.margin_mae_m != null ? surveyed.margin_mae_m.toFixed(3) : "—"), unit: "m",
      k: "Mean margin error", n: "per frame, against the exact driven margin",
    },
    {
      v: medianPeak != null ? medianPeak.toFixed(3) : "—", unit: "m",
      k: "Median peak-margin error", n: "across the four graduated excursions",
    },
    {
      v: attributed ? pct(attrCorrect / attributed, 0) : "—", k: "Correct car named",
      n: `${attrCorrect} of ${attributed} tracks attributed`,
    },
  ].map((t) => `<div class="tile"><div class="v">${t.v}${t.unit ? `<small>${t.unit}</small>` : ""}</div>
      <div class="k">${esc(t.k)}</div><div class="n">${esc(t.n)}</div></div>`).join("");

  /* --------------------------------------------------------------- queue */
  const modes = [...new Set(DATA.cases.map((c) => c.calibration_mode))];
  const state = { mode: modes.includes("surveyed") ? "surveyed" : modes[0], onlyHeldOut: false, id: null, car: null };

  const filters = [
    ...modes.map((mo) => ({ key: "mode", value: mo, label: mo === "surveyed" ? "Surveyed camera" : "Self-calibrated" })),
    { key: "held", value: true, label: "Unseen camera only" },
  ];
  $("qfilters").innerHTML = filters.map((f, i) =>
    `<button type="button" data-i="${i}" aria-pressed="false">${esc(f.label)}</button>`).join("");

  function visibleCases() {
    return DATA.cases.filter((c) => c.calibration_mode === state.mode
      && (!state.onlyHeldOut || c.held_out_camera));
  }
  function verdictClass(c) {
    if (c.verdict === "offence") return "offence";
    if (c.verdict === "no offence") return "clear";
    return "flag";
  }
  function verdictLabel(c) {
    if (c.verdict === "offence") return "Offence reported";
    if (c.verdict === "no offence") return "No offence";
    if (c.verdict === "missed") return "Offence not reported";
    return "Unmatched report";
  }

  function renderQueue() {
    const rows = visibleCases();
    if (!rows.some((c) => c.id === state.id)) { state.id = rows.length ? rows[0].id : null; state.car = null; }
    $("qlist").innerHTML = rows.map((c) => `
      <button type="button" role="listitem" class="qrow ${verdictClass(c)}" data-id="${esc(c.id)}"
              aria-current="${c.id === state.id}">
        <i class="stripe"></i>
        <span class="body">
          <span class="name">${esc(pretty(c.scenario))}</span>
          <span class="sub">${esc(c.camera)}${c.held_out_camera ? " · unseen" : ""}</span>
        </span>
        <span class="right">
          <span class="marg">${c.peak_margin_detected != null ? m(c.peak_margin_detected) + " m" : "—"}</span>
          <span class="conf"><i style="width:${Math.round((c.confidence || 0) * 100)}%"></i></span>
        </span>
      </button>`).join("");
    $("qlist").querySelectorAll(".qrow").forEach((b) => b.addEventListener("click", () => {
      state.id = b.dataset.id; state.car = null; renderQueue(); renderDetail();
    }));
  }

  $("qfilters").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    const f = filters[+b.dataset.i];
    if (f.key === "mode") state.mode = f.value; else state.onlyHeldOut = !state.onlyHeldOut;
    syncFilters(); renderQueue(); renderDetail();
  }));
  function syncFilters() {
    $("qfilters").querySelectorAll("button").forEach((b) => {
      const f = filters[+b.dataset.i];
      b.setAttribute("aria-pressed", f.key === "mode" ? String(state.mode === f.value) : String(state.onlyHeldOut));
    });
  }

  /* -------------------------------------------------------- margin chart */
  function marginChart(trace, host) {
    const W = 920, H = 300, L = 56, R = 96, T = 16, B = 34;
    const frames = trace.frame;
    const values = trace.detected.concat(trace.truth);
    let lo = Math.min(...values), hi = Math.max(...values, 0.05);
    const pad = Math.max((hi - lo) * 0.14, 0.05);
    lo -= pad; hi += pad;
    const x = (f) => L + (f - frames[0]) / Math.max(frames[frames.length - 1] - frames[0], 1) * (W - L - R);
    const y = (v) => T + (hi - v) / (hi - lo) * (H - T - B);

    const box = document.createElement("div");
    box.className = "chartbox";
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img" });
    svg.style.height = "auto";
    svg.setAttribute("aria-label", "Detected track-limit margin compared with the exact driven margin, by frame");
    box.appendChild(svg);

    // shaded truth excursion window
    let run = null;
    trace.true_violation.forEach((v, i) => {
      if (v && run === null) run = i;
      if ((!v || i === frames.length - 1) && run !== null) {
        const end = v ? i : i - 1;
        el("rect", {
          x: x(frames[run]) - 1, y: T, width: Math.max(x(frames[end]) - x(frames[run]) + 2, 2),
          height: H - T - B, fill: "var(--truth)", opacity: ".10",
        }, svg);
        run = null;
      }
    });

    // grid + y axis
    const ticks = 5;
    for (let i = 0; i < ticks; i++) {
      const v = lo + (hi - lo) * i / (ticks - 1);
      el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "var(--rule)", "stroke-width": 1 }, svg);
      const t = el("text", { x: L - 9, y: y(v) + 4, "text-anchor": "end", fill: "var(--ink-3)",
        "font-size": 11, "font-family": "var(--mono)" }, svg);
      t.textContent = v.toFixed(2);
    }
    // x axis ticks
    const span = frames[frames.length - 1] - frames[0];
    const stepF = span > 60 ? 16 : 8;
    for (let f = frames[0]; f <= frames[frames.length - 1]; f += stepF) {
      const t = el("text", { x: x(f), y: H - 12, "text-anchor": "middle", fill: "var(--ink-3)",
        "font-size": 11, "font-family": "var(--mono)" }, svg);
      t.textContent = f;
    }
    const xl = el("text", { x: (L + W - R) / 2, y: H - 1, "text-anchor": "middle", fill: "var(--ink-3)", "font-size": 11 }, svg);
    xl.textContent = "frame";

    // the track limit itself
    el("line", { x1: L, x2: W - R, y1: y(0), y2: y(0), stroke: "var(--accent)", "stroke-width": 1.5 }, svg);
    const lim = el("text", { x: W - R + 8, y: y(0) + 4, fill: "var(--accent)", "font-size": 11.5,
      "font-family": "var(--cond)", "letter-spacing": ".08em" }, svg);
    lim.textContent = "TRACK LIMIT";

    const path = (vals, stroke, dash) => {
      const d = vals.map((v, i) => `${i ? "L" : "M"}${x(frames[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
      el("path", { d, fill: "none", stroke, "stroke-width": 2, "stroke-linejoin": "round",
        "stroke-linecap": "round", ...(dash ? { "stroke-dasharray": dash } : {}) }, svg);
    };
    path(trace.truth, "var(--truth)", "5 4");
    path(trace.detected, "var(--detected)");

    // direct label on the detected peak
    let pk = 0;
    trace.detected.forEach((v, i) => { if (v > trace.detected[pk]) pk = i; });
    el("circle", { cx: x(frames[pk]), cy: y(trace.detected[pk]), r: 4.5, fill: "var(--detected)",
      stroke: "var(--surface)", "stroke-width": 2 }, svg);
    const lab = el("text", { x: x(frames[pk]) + 9, y: y(trace.detected[pk]) - 8, fill: "var(--ink)",
      "font-size": 11.5, "font-family": "var(--mono)" }, svg);
    lab.textContent = `peak ${m(trace.detected[pk])} m`;

    // hover layer
    const cross = el("line", { x1: 0, x2: 0, y1: T, y2: H - B, stroke: "var(--rule-strong)",
      "stroke-width": 1, opacity: 0 }, svg);
    const tip = document.createElement("div");
    tip.className = "tooltip";
    box.appendChild(tip);
    const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, svg);
    hit.style.cursor = "crosshair";
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const px = (ev.clientX - r.left) / r.width * W;
      let i = Math.round((px - L) / (W - L - R) * span);
      i = Math.max(0, Math.min(frames.length - 1, i));
      cross.setAttribute("x1", x(frames[i])); cross.setAttribute("x2", x(frames[i]));
      cross.setAttribute("opacity", 1);
      tip.style.opacity = 1;
      tip.innerHTML = `frame ${frames[i]}<br>detected&nbsp;<b>${m(trace.detected[i])} m</b>`
        + `<br>driven&nbsp;&nbsp;&nbsp;<b>${m(trace.truth[i])} m</b>`
        + `<br>error&nbsp;&nbsp;&nbsp;&nbsp;${m(trace.detected[i] - trace.truth[i])} m`;
      const px2 = x(frames[i]) / W * r.width;
      tip.style.left = Math.min(Math.max(px2 + 12, 4), r.width - tip.offsetWidth - 4) + "px";
      tip.style.top = "10px";
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); tip.style.opacity = 0; });
    host.appendChild(box);
  }

  /* -------------------------------------------------------------- detail */
  function renderDetail() {
    const c = DATA.cases.find((k) => k.id === state.id);
    const host = $("detail");
    if (!c) { host.innerHTML = `<div class="pane"><p>No case matches these filters.</p></div>`; return; }
    const cars = Object.keys(c.trace);
    if (!state.car || !cars.includes(state.car)) {
      const offender = c.attribution.tracks.find((t) => t.claimed);
      state.car = cars.includes(offender?.claimed) ? offender.claimed : cars[0];
    }
    const trace = c.trace[state.car];
    const ev = c.event_detected, tr = c.event_truth;
    const cls = verdictClass(c);
    const attr = c.attribution.tracks.find((t) => t.claimed === state.car) || c.attribution.tracks[0] || {};
    const conf = ev ? (ev.confidence || 0) : 0;

    host.innerHTML = `
      <div class="verdict">
        <div class="head">
          <div class="eyebrow">${esc(c.camera)} camera &middot; ${esc(c.calibration_mode === "surveyed" ? "surveyed calibration" : "self-calibrated")}${c.held_out_camera ? " &middot; never trained on" : ""}</div>
          <h3>${esc(pretty(c.scenario))}</h3>
        </div>
        <div><span class="badge ${cls}"><i class="dot"></i>${esc(verdictLabel(c))}</span></div>
      </div>
      <div class="facts">
        <div class="fact"><div class="k">Peak margin</div><div class="v">${ev ? m(ev.peak_margin_m) : "—"}</div></div>
        <div class="fact"><div class="k">Actually driven</div><div class="v">${tr ? m(tr.peak_margin_m) : (c.peak_margin_true != null ? m(c.peak_margin_true) : "—")}</div></div>
        <div class="fact"><div class="k">Error</div><div class="v ${ev && tr ? (Math.abs(ev.peak_margin_m - tr.peak_margin_m) < 0.05 ? "good" : "") : ""}">${ev && tr ? m(ev.peak_margin_m - tr.peak_margin_m) : "—"}</div></div>
        <div class="fact"><div class="k">Frames off track</div><div class="v">${ev ? ev.frame_count : 0}${tr ? ` <small style="color:var(--ink-3)">/ ${tr.frame_count}</small>` : ""}</div></div>
        <div class="fact"><div class="k">Confidence</div><div class="v">${ev ? pct(conf, 0) : "—"}</div></div>
        <div class="fact"><div class="k">Car named</div><div class="v sm">${esc(attr.claimed || "unattributed")}${attr.claimed ? (attr.correct ? ' <span class="good">✓</span>' : ' <span class="bad">✗</span>') : ""}</div></div>
      </div>
      <div class="plot">
        <div class="plot-head">
          <h4>Margin against the driven truth${cars.length > 1 ? ` &mdash; ${esc(state.car)}` : ""}</h4>
          <div class="legend">
            <span><i style="border-color:var(--detected)"></i>pipeline</span>
            <span><i style="border-color:var(--truth);border-top-style:dashed"></i>driven</span>
            <span><i style="border-color:var(--accent)"></i>track limit</span>
            <span style="color:var(--ink-3)">shaded: frames actually off track</span>
          </div>
        </div>
        ${cars.length > 1 ? `<div class="qfilters" style="border:0;padding:0 0 8px" id="carpick">${cars.map((k) =>
          `<button type="button" data-car="${esc(k)}" aria-pressed="${k === state.car}">${esc(k)}</button>`).join("")}</div>` : ""}
        <figure id="plot-host"></figure>
        <figcaption>Positive is beyond the outer edge of the white line. A frame counts as an offence only when all four tyre contact points are positive; the margin plotted is the least-outside corner.</figcaption>
      </div>
      <div class="strip">
        <div class="striprow"><span>Driven</span><div class="cells">${trace.true_violation.map((v) => `<i class="${v ? "on" : ""}"></i>`).join("")}</div></div>
        <div class="striprow"><span>Pipeline</span><div class="cells">${trace.detected_violation.map((v) => `<i class="${v ? "det" : ""}"></i>`).join("")}</div></div>
      </div>
      <div class="split">
        <div class="pane">
          <h4>Why this confidence</h4>
          ${ev ? confidenceBars(c, conf) : `<p style="color:var(--ink-2);font-size:13.5px;margin:0">No sustained excursion was reported for this clip.</p>`}
        </div>
        <div class="pane">
          <h4>Timing and identity</h4>
          <dl class="kv">
            <dt>Start of excursion</dt><dd>${ev ? `frame ${ev.start_frame}` : "—"}${tr ? ` <span style="color:var(--ink-3)">(driven ${tr.start_frame})</span>` : ""}</dd>
            <dt>Duration error</dt><dd>${c.duration_error_frames != null ? `${c.duration_error_frames > 0 ? "+" : ""}${c.duration_error_frames} frames` : "—"}</dd>
            <dt>Attribution rung</dt><dd>${esc(attr.rung || "—")}</dd>
            <dt>Detection recall</dt><dd>${pct(c.detection.recall)}</dd>
            <dt>Contact-point error</dt><dd>${c.detection.mean_contact_error_m != null ? c.detection.mean_contact_error_m.toFixed(3) + " m" : "—"}</dd>
            <dt>Per-frame margin error</dt><dd>${c.margin.margin_mae_m != null ? c.margin.margin_mae_m.toFixed(3) + " m" : "—"}</dd>
          </dl>
          ${suppressedNote(c)}
        </div>
      </div>`;

    marginChart(trace, $("plot-host"));
    const pick = $("carpick");
    if (pick) pick.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      state.car = b.dataset.car; renderDetail();
    }));
  }

  function suppressedNote(c) {
    if (c.correctly_suppressed_events && c.correctly_suppressed_events.length) {
      const s = c.correctly_suppressed_events[0].truth;
      return `<div class="note">A ${s.frame_count}-frame excursion by ${esc(c.correctly_suppressed_events[0].car_id)} was
        held below the ${c.sustained_min_frames}-frame sustained threshold and deliberately not reported.</div>`;
    }
    if (c.missed_events && c.missed_events.length) {
      const s = c.missed_events[0].truth;
      return `<div class="note">A ${s.frame_count}-frame excursion by ${esc(c.missed_events[0].car_id)}
        peaking at ${m(s.peak_margin_m)} m was not reported.</div>`;
    }
    return "";
  }

  function confidenceBars(c, conf) {
    const rows = [["Reported confidence", conf]];
    return `<div class="bars">${rows.map(([k, v]) =>
      `<div class="barrow"><span>${esc(k)}</span><span class="t"><i style="width:${Math.round(v * 100)}%"></i></span><span class="n">${pct(v, 0)}</span></div>`
    ).join("")}</div>
    <p style="margin:10px 0 0;font-size:13px;color:var(--ink-2)">The score is the probability that the peak margin is genuinely above zero,
    given the frame-to-frame measurement noise the clip itself reveals, discounted for a short run,
    a weak detection or a track that kept dropping out.</p>`;
  }

  /* ---------------------------------------------------- graduated chart */
  function graduated() {
    const pts = DATA.graduated.filter((g) => g.calibration_mode === "surveyed");
    const host = $("grad-card");
    if (!pts.length) { host.innerHTML = "<h3>Graduated excursions</h3><p>No graduated cases in this run.</p>"; return; }
    const cams = [...new Set(pts.map((p) => p.camera))];
    const colour = (cam) => `var(--cam-${cams.indexOf(cam) + 1})`;
    const W = 460, H = 340, L = 58, R = 18, T = 16, B = 46;
    const vals = pts.flatMap((p) => [p.detected_peak_m, p.target_m]);
    let lo = Math.min(0, ...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
    const x = (v) => L + (v - lo) / (hi - lo) * (W - L - R);
    const y = (v) => T + (hi - v) / (hi - lo) * (H - T - B);

    host.innerHTML = `<h3>Reported peak against commanded peak</h3>
      <p>Each point is one clip. The diagonal is a perfect reading.</p>
      <div class="legend" style="margin-bottom:8px">${cams.map((cm) =>
        `<span><i style="border-color:${colour(cm)}"></i>${esc(cm)}</span>`).join("")}</div>`;
    const box = document.createElement("div");
    box.className = "chartbox";
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
      "aria-label": "Reported peak margin against commanded peak margin for each graduated excursion" });
    svg.style.height = "auto";
    box.appendChild(svg);

    for (let i = 0; i <= 4; i++) {
      const v = lo + (hi - lo) * i / 4;
      el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "var(--rule)", "stroke-width": 1 }, svg);
      const t = el("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", fill: "var(--ink-3)", "font-size": 11,
        "font-family": "var(--mono)" }, svg);
      t.textContent = v.toFixed(2);
      const b = el("text", { x: x(v), y: H - 26, "text-anchor": "middle", fill: "var(--ink-3)", "font-size": 11,
        "font-family": "var(--mono)" }, svg);
      b.textContent = v.toFixed(2);
    }
    el("line", { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), stroke: "var(--rule-strong)", "stroke-width": 1.5 }, svg);
    const xt = el("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle", fill: "var(--ink-3)", "font-size": 11.5 }, svg);
    xt.textContent = "commanded peak margin (m)";
    const yt = el("text", { x: 14, y: (T + H - B) / 2, "text-anchor": "middle", fill: "var(--ink-3)", "font-size": 11.5,
      transform: `rotate(-90 14 ${(T + H - B) / 2})` }, svg);
    yt.textContent = "reported peak margin (m)";

    const tip = document.createElement("div");
    tip.className = "tooltip";
    box.appendChild(tip);
    pts.forEach((p) => {
      const cx = x(p.target_m), cy = y(p.detected_peak_m);
      const c = el("circle", { cx, cy, r: 5.5, fill: colour(p.camera), stroke: "var(--surface)", "stroke-width": 2 }, svg);
      c.style.cursor = "pointer";
      c.addEventListener("pointerenter", () => {
        tip.style.opacity = 1;
        tip.innerHTML = `${esc(p.camera)}${p.held_out_camera ? " (unseen)" : ""}<br>commanded <b>${p.target_m.toFixed(2)} m</b>`
          + `<br>reported&nbsp; <b>${m(p.detected_peak_m)} m</b><br>error&nbsp;&nbsp;&nbsp;&nbsp; ${m(p.detected_peak_m - p.target_m)} m`;
        const r = svg.getBoundingClientRect();
        tip.style.left = Math.min(cx / W * r.width + 12, r.width - 150) + "px";
        tip.style.top = Math.max(cy / H * r.height - 20, 4) + "px";
      });
      c.addEventListener("pointerleave", () => { tip.style.opacity = 0; });
    });
    host.appendChild(box);

    const byTarget = {};
    pts.forEach((p) => { (byTarget[p.target_m] ||= []).push(p); });
    $("grad-table").innerHTML = `<h3>Per-excursion readings</h3>
      <p>Reported peak margin from each camera, against the exact value the simulator drove.</p>
      <div class="tablewrap" style="border:0"><table>
      <thead><tr><th>Commanded</th>${cams.map((cm) => `<th>${esc(cm)}</th>`).join("")}<th>spread</th></tr></thead>
      <tbody>${Object.keys(byTarget).sort((a, b) => a - b).map((t) => {
        const row = byTarget[t];
        const got = cams.map((cm) => row.find((p) => p.camera === cm));
        const errs = got.filter(Boolean).map((p) => p.detected_peak_m);
        const spread = errs.length > 1 ? (Math.max(...errs) - Math.min(...errs)) : null;
        return `<tr><td>${(+t).toFixed(2)} m</td>${got.map((p) =>
          `<td>${p ? m(p.detected_peak_m) : "—"}</td>`).join("")}<td>${spread != null ? spread.toFixed(3) : "—"}</td></tr>`;
      }).join("")}</tbody></table></div>`;
  }

  /* -------------------------------------------------------- calibration */
  function calibration() {
    const rows = DATA.calibration;
    if (!rows.length) { $("cal-table").innerHTML = ""; return; }
    $("cal-table").innerHTML = `<table>
      <thead><tr><th>Camera</th><th>Position error</th><th>Field of view</th><th>Lateral error</th>
        <th>p95 lateral</th><th>Along-track offset</th><th>Solve time</th></tr></thead>
      <tbody>${rows.map((r) => `
        <tr class="${r.held_out_camera ? "marked" : ""}">
          <td>${esc(r.camera)}</td>
          <td>${r.position_error_m.toFixed(2)} m</td>
          <td>${r.fov_estimated_deg.toFixed(2)}° <span style="color:var(--ink-3)">/ ${r.fov_true_deg}°</span></td>
          <td class="${r.lateral_error_m < 0.5 ? "good" : "bad"}">${r.lateral_error_m.toFixed(3)} m</td>
          <td>${r.lateral_p95_m.toFixed(3)} m</td>
          <td>${m(r.along_track_gauge_m, 2)} m</td>
          <td>${r.seconds != null ? r.seconds.toFixed(0) + " s" : "—"}</td>
        </tr>`).join("")}</tbody></table>`;
  }

  /* ------------------------------------------------------------ baseline */
  function baseline() {
    const b = DATA.baseline;
    const host = $("baseline-card");
    if (!b) { host.innerHTML = `<h3>Off-the-shelf baseline</h3><p>Not run.</p>`; return; }
    host.innerHTML = `<h3>What an off-the-shelf detector sees</h3>
      <p>${esc(b.weights)} with COCO weights and no fine-tuning, on the ${esc(b.camera)} camera.</p>
      <dl class="kv">
        <dt>Cars matched as a vehicle class</dt><dd>${b.matched} / ${b.cars}</dd>
        <dt>Vehicle-class recall</dt><dd class="${b.vehicle_class_recall > 0.5 ? "good" : "bad"}">${pct(b.vehicle_class_recall)}</dd>
        <dt>Confidence floor used</dt><dd>${b.confidence_threshold}</dd>
      </dl>
      <div class="labels">${b.top_predicted_labels.slice(0, 8).map(([k, v]) =>
        `<span><b>${esc(k)}</b> ${v}</span>`).join("")}</div>
      <div class="note">${esc(b.note)}</div>`;
  }

  /* ------------------------------------------------------------ training */
  function training() {
    const log = DATA.training;
    const host = $("training-card");
    if (!log || log.length < 4) { host.innerHTML = `<h3>Training</h3><p>No log recorded.</p>`; return; }
    const W = 420, H = 170, L = 46, R = 12, T = 12, B = 30;
    const hi = Math.max(...log.map((r) => r.loss)), lo = 0;
    const x = (i) => L + i / (log.length - 1) * (W - L - R);
    const y = (v) => T + (hi - v) / (hi - lo) * (H - T - B);
    host.innerHTML = `<h3>Training on this footage</h3>
      <p>${log[log.length - 1].step} iterations over ${meta.train_cameras.join(" + ")} and the moving rig.
      The ${esc(meta.held_out_camera)} camera and ${meta.held_out_scenarios.map(pretty).join(", ")} were withheld.</p>`;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
      "aria-label": "Training loss against iteration" });
    svg.style.height = "auto";
    for (let i = 0; i <= 3; i++) {
      const v = lo + (hi - lo) * i / 3;
      el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "var(--rule)", "stroke-width": 1 }, svg);
      const t = el("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", fill: "var(--ink-3)", "font-size": 10.5,
        "font-family": "var(--mono)" }, svg);
      t.textContent = v.toFixed(1);
    }
    el("path", { d: log.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(r.loss).toFixed(1)}`).join(""),
      fill: "none", stroke: "var(--detected)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
    const last = el("circle", { cx: x(log.length - 1), cy: y(log[log.length - 1].loss), r: 4,
      fill: "var(--detected)", stroke: "var(--surface)", "stroke-width": 2 }, svg);
    const t1 = el("text", { x: L, y: H - 8, fill: "var(--ink-3)", "font-size": 10.5, "font-family": "var(--mono)" }, svg);
    t1.textContent = "step " + log[0].step;
    const t2 = el("text", { x: W - R, y: H - 8, "text-anchor": "end", fill: "var(--ink-3)", "font-size": 10.5,
      "font-family": "var(--mono)" }, svg);
    t2.textContent = "step " + log[log.length - 1].step;
    host.appendChild(svg);
  }

  $("footer").innerHTML = `Judgements are produced from the video, the circuit survey and a ground homography only;
    the simulator's per-frame geometry is used to train the detector on the held-in clips and to score this page afterwards.
    Positive margins are metres beyond the outer edge of the white line. Generated ${esc(meta.generated_utc)}.`;

  syncFilters(); renderQueue(); renderDetail(); graduated(); calibration(); baseline(); training();
})();
