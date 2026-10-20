"use strict";
/*
 * Human eval: accent similarity, accentedness, naturalness (MOS), voice similarity.
 * Static page, no dependencies. Responses are kept in the browser, offered as a JSON download and
 * optionally POSTed to CONFIG.SUBMIT_URL (e.g. a Google Apps Script web app or your own endpoint).
 *
 * URL parameters:
 *   r=<int>      rater index; defines the Latin-square position (system for item i = SYSTEMS[(i + r) % 5]).
 *                Hand out r = 0,1,2,3,4,0,1,... so that all 5 positions are used equally often.
 *   pid=<string> external participant id (e.g. Prolific PID), stored with the answers.
 *   dev=1        relaxes consent / "listen fully" requirements for testing. NEVER for real participants.
 */
const CONFIG = {
  SUBMIT_URL: "",                  // optional endpoint; leave empty to only download the JSON
  STUDY_ID: "accent-ot-v1",
  MIN_ANCHORS_PLAYED: 2,           // reference clips that must be played before a block starts
};

const params = new URLSearchParams(location.search);
const DEV = params.get("dev") === "1";
const PID = params.get("pid") || "";

const QUESTIONS = [
  { id: "accent_sim", n: 7,
    text: (L) => `How similar is this speaker's accent to the reference accent (${L})?`,
    help: "Judge only the accent (pronunciation, rhythm, intonation), not the voice or recording quality.",
    ends: ["Not at all similar", "Very similar"] },
  { id: "accentedness", n: 9,
    text: () => "How strong is this speaker's non-native accent?",
    help: "1 = sounds like a native speaker of English, 9 = very strong foreign accent.",
    ends: ["No accent (native-like)", "Extremely strong accent"] },
  { id: "mos", n: 5,
    text: () => "How natural and clear does this recording sound?",
    help: "Judge the overall speech quality and naturalness. Ignore the accent.",
    verbal: ["Bad", "Poor", "Fair", "Good", "Excellent"] },
  { id: "voice_sim", n: 5,
    text: () => "Compare with the voice reference. Is it the same person speaking?",
    help: "Ignore accent, content and recording quality; focus on the voice (timbre, pitch).",
    verbal: ["Definitely a different person", "Probably different", "Not sure", "Probably the same", "Definitely the same person"] },
];

// ── helpers ───────────────────────────────────────────────────────────────────
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(kid));
  return e;
}
function mulberry32(a) {
  return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function shuffle(arr, rng) { for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; }
function hashStr(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
const $app = document.getElementById("app");
const $progress = document.getElementById("progress");
function show(...nodes) { $app.replaceChildren(...nodes); window.scrollTo(0, 0); }

function player(label, src, { required = false, onChange } = {}) {
  const st = { plays: 0, ended: false };
  const audio = el("audio", { controls: true, preload: "auto", src, controlslist: "nodownload noplaybackrate" });
  const status = el("span", { class: "status" });
  const refresh = (notify = true) => {
    if (required && !DEV) { status.textContent = st.ended ? "✓ listened" : "Please listen to the whole recording"; status.className = "status " + (st.ended ? "ok" : "wait"); }
    else status.textContent = "";
    if (notify && onChange) onChange(st);   // no callback at construction: callers' state is not ready yet
  };
  audio.addEventListener("play", () => { st.plays++; refresh(); });
  audio.addEventListener("ended", () => { st.ended = true; refresh(); });
  audio.addEventListener("error", () => { status.textContent = "Audio failed to load. Please reload the page."; status.className = "status wait"; });
  refresh(false);
  return { st, node: el("div", { class: "player" }, el("div", { class: "tag" }, label), audio, status), audio };
}
function stopAll() { document.querySelectorAll("audio").forEach(a => a.pause()); }

// ── state ─────────────────────────────────────────────────────────────────────
let M, R, STATE;
const storeKey = () => `he_${CONFIG.STUDY_ID}_r${R}_${PID}`;
function save() { try { localStorage.setItem(storeKey(), JSON.stringify(STATE)); } catch (e) { /* ignore */ } }
function load() { try { return JSON.parse(localStorage.getItem(storeKey())); } catch (e) { return null; } }

function buildBlocks(m, r, rng) {
  const pos = r % m.n_systems;
  const order = shuffle(m.accent_order.slice(), rng);
  const checkTgts = shuffle(m.accent_order.slice(), rng).slice(0, 2);
  const repeatTgts = shuffle(m.accent_order.slice(), rng).slice(0, 2);
  return order.map((tgt) => {
    const trials = shuffle(m.items.filter(i => i.tgt === tgt), rng).map(it =>
      ({ kind: "item", item_id: it.id, stimulus: it.stimuli[pos], voice_ref: it.voice_ref }));
    if (checkTgts.includes(tgt)) {
      const c = m.checks.find(c => c.tgt === tgt);
      trials.splice(1 + Math.floor(rng() * (trials.length)), 0,
        { kind: "check", item_id: c.id, stimulus: c.stimulus, voice_ref: c.voice_ref });
    }
    if (repeatTgts.includes(tgt)) trials.push({ ...trials.find(t => t.kind === "item"), kind: "repeat" });
    return { tgt, trials };
  });
}

// ── screens ───────────────────────────────────────────────────────────────────
function screenIntro() {
  const consent = el("input", { type: "checkbox", id: "consent" });
  const next = el("button", { class: "btn", disabled: !DEV, onclick: screenSetup }, "Continue");
  consent.addEventListener("change", () => { next.disabled = !consent.checked && !DEV; });
  show(
    el("h1", {}, "Listening test: accent, quality and voice"),
    el("p", {}, "You will listen to short recordings of English sentences (2–6 seconds each) and rate them. Some are natural recordings, others are processed by computer algorithms. There are no right or wrong answers; we are interested in your honest impression."),
    el("div", { class: "card" },
      el("p", {}, el("b", {}, "What to expect")),
      el("ul", {},
        el("li", {}, "About 25 minutes, 6 short blocks, about 28 recordings in total."),
        el("li", {}, "Please use headphones in a quiet place and set a comfortable volume before you start."),
        el("li", {}, "Each recording must be played to the end before you can rate it. You may replay it."),
        el("li", {}, "You may stop at any time by closing the page; partial data is only kept in your browser unless you submit."))),
    el("div", { class: "card" },
      el("p", {}, el("b", {}, "Data and consent")),
      el("p", { class: "muted" }, "We collect your ratings, response times and the optional background questions at the end (no name, e-mail or IP address is requested by this page). Data is used for research on accent processing and may be published in anonymised, aggregated form. Participation is voluntary and you can withdraw at any time before submitting. Contact the research team for questions (see the invitation)."),
      el("label", {}, consent, " I am at least 18 years old, I have read the information above and I agree to take part.")),
    next);
}

function screenSetup() {
  const dev = el("select", { id: "device" },
    ...[["", "Select…"], ["headphones", "Headphones (over-ear)"], ["earbuds", "Earbuds / in-ear"], ["speakers", "Loudspeakers"], ["other", "Other"]]
      .map(([v, t]) => el("option", { value: v }, t)));
  const heard = el("select", { id: "heard" },
    ...[["", "Select…"], ["yes", "Yes, clearly"], ["no", "No / unclear"]].map(([v, t]) => el("option", { value: v }, t)));
  const quiet = el("input", { type: "checkbox", id: "quiet" });
  const next = el("button", { class: "btn", disabled: true, onclick: () => { STATE.setup = { device: dev.value, quiet: quiet.checked, heard: heard.value }; save(); screenInstructions(); } }, "Continue");
  const p = player("Calibration clip: please play it and adjust your volume", M.items[0].voice_ref, { required: true, onChange: check });
  function check() { next.disabled = !DEV && !(p.st.ended && dev.value && heard.value === "yes" && quiet.checked); }
  [dev, heard, quiet].forEach(x => x.addEventListener("change", check));
  check();
  show(el("h1", {}, "Audio check"),
    p.node,
    el("label", { class: "field" }, "What are you listening with? ", dev),
    el("label", { class: "field" }, "Could you hear the clip clearly? ", heard),
    el("label", { class: "field" }, quiet, " I am in a quiet place and will not be disturbed."),
    el("p", { class: "muted" }, "If you could not hear the clip clearly, check your sound settings and reload the page."),
    next);
}

function screenInstructions() {
  show(el("h1", {}, "How the test works"),
    el("p", {}, "The test has 6 blocks. At the start of each block you hear three reference clips of one English accent (for example, Korean-accented English). Then you rate several recordings. For each recording you answer four questions:"),
    el("div", { class: "card" }, el("ol", {},
      el("li", {}, el("b", {}, "Accent similarity (1–7): "), "how similar is the accent to the block's reference accent?"),
      el("li", {}, el("b", {}, "Accent strength (1–9): "), "how strong is the non-native accent in general? 1 = like a native speaker."),
      el("li", {}, el("b", {}, "Quality (1–5): "), "how natural and clear is the recording? Ignore the accent."),
      el("li", {}, el("b", {}, "Voice (1–5): "), "compared with a reference recording of the original speaker, is it the same person? Ignore accent and quality."))),
    el("p", {}, "The recordings may sound unusual: some are synthetic, noisy or distorted. Please rate what you hear. Use the full range of each scale."),
    el("button", { class: "btn", onclick: () => nextBlock() }, "Start"));
}

function nextBlock() {
  stopAll();
  if (STATE.block >= STATE.blocks.length) return screenDemographics();
  screenBlockIntro();
}

function screenBlockIntro() {
  const b = STATE.blocks[STATE.block];
  const label = M.accent_labels[b.tgt];
  let ended = 0;
  const start = el("button", { class: "btn", disabled: !DEV, onclick: () => { stopAll(); STATE.trial = 0; save(); screenTrial(); } }, "Start block");
  const ps = M.anchors[b.tgt].map((src, i) => player(`Reference ${i + 1} (${label})`, src, {
    required: false, onChange: () => { ended = ps.filter(p => p.st.ended).length; start.disabled = !DEV && ended < CONFIG.MIN_ANCHORS_PLAYED; hint.textContent = ended >= CONFIG.MIN_ANCHORS_PLAYED || DEV ? "" : `Please play at least ${CONFIG.MIN_ANCHORS_PLAYED} reference clips to the end.`; } }));
  const hint = el("p", { class: "status wait" });
  $progress.hidden = false; $progress.textContent = `Block ${STATE.block + 1} of ${STATE.blocks.length}`;
  show(el("h1", {}, `Block ${STATE.block + 1} of ${STATE.blocks.length}: ${label}`),
    el("p", {}, `Listen to these examples of ${label}. In this block, you will judge how similar each recording's accent is to this reference accent.`),
    ...ps.map(p => p.node), hint, start);
}

function screenTrial() {
  const b = STATE.blocks[STATE.block];
  const t = b.trials[STATE.trial];
  const label = M.accent_labels[b.tgt];
  const total = STATE.blocks.reduce((n, x) => n + x.trials.length, 0);
  const done = STATE.blocks.slice(0, STATE.block).reduce((n, x) => n + x.trials.length, 0) + STATE.trial;
  $progress.hidden = false;
  $progress.textContent = `Block ${STATE.block + 1}/${STATE.blocks.length} · recording ${done + 1}/${total}`;
  const t0 = performance.now();
  const answers = {};
  const next = el("button", { class: "btn", disabled: true, onclick: submit }, "Next");
  const qs = el("div", {});
  const main = player("Recording to rate", t.stimulus, { required: true, onChange: refresh });
  const voice = player("Voice reference: original recording of this speaker (different sentence)", t.voice_ref, {});
  const anchors = el("details", {}, el("summary", {}, `Reference accent clips (${label})`),
    ...M.anchors[b.tgt].map((s, i) => player(`Reference ${i + 1}`, s, {}).node));
  const fieldsets = QUESTIONS.map(q => {
    const name = `q_${q.id}`;
    const opts = Array.from({ length: q.n }, (_, i) => i + 1).map(v => el("label", {},
      el("input", { type: "radio", name, value: v, onchange: () => { answers[q.id] = v; refresh(); } }),
      el("span", {}, q.verbal ? `${v} · ${q.verbal[v - 1]}` : String(v))));
    return el("fieldset", { class: "q", disabled: !DEV }, el("legend", {}, q.text(label)),
      el("div", { class: "muted" }, q.help),
      el("div", { class: "scale" + (q.verbal ? " verbal" : "") }, opts),
      q.ends ? el("div", { class: "ends" }, el("span", {}, `1 = ${q.ends[0]}`), el("span", {}, `${q.n} = ${q.ends[1]}`)) : null);
  });
  qs.append(...fieldsets);
  function refresh() {
    const ok = DEV || main.st.ended;
    fieldsets.forEach(f => { f.disabled = !ok; });
    next.disabled = !(ok && QUESTIONS.every(q => answers[q.id] != null));
  }
  function submit() {
    stopAll();
    STATE.responses.push({
      trial_index: STATE.responses.length, block_index: STATE.block, tgt: b.tgt, kind: t.kind, item_id: t.item_id,
      stimulus: t.stimulus, voice_ref: t.voice_ref, ratings: { ...answers },
      plays: main.st.plays, voice_ref_plays: voice.st.plays, listened_fully: main.st.ended,
      rt_ms: Math.round(performance.now() - t0), t_end: new Date().toISOString(),
    });
    STATE.trial++;
    if (STATE.trial >= b.trials.length) { STATE.block++; STATE.trial = 0; save(); nextBlock(); }
    else { save(); screenTrial(); }
  }
  show(main.node, voice.node, anchors, qs, next);
}

function screenDemographics() {
  $progress.hidden = true;
  const sel = (id, opts) => el("select", { id }, ...[["", "Prefer not to say"], ...opts.map(o => [o, o])].map(([v, t]) => el("option", { value: v }, t)));
  const age = sel("age", ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"]);
  const nat = el("input", { type: "text", id: "native", placeholder: "e.g. English, Russian" });
  const prof = sel("prof", ["Native / bilingual", "Fluent (C1–C2)", "Upper-intermediate (B2)", "Intermediate or lower"]);
  const fam = {};
  const famRows = M.accent_order.map(a => {
    const s = el("select", {}, ...[["", "—"], ["1", "Never heard"], ["2", "Rarely"], ["3", "Sometimes"], ["4", "Often"], ["5", "Very familiar"]].map(([v, t]) => el("option", { value: v }, t)));
    fam[a] = s; return el("tr", {}, el("td", {}, M.accent_labels[a]), el("td", {}, s));
  });
  const comment = el("textarea", { rows: 3, style: "width:100%", placeholder: "Anything unusual about the recordings or the test? (optional)" });
  show(el("h1", {}, "Almost done"), el("p", { class: "muted" }, "All questions below are optional."),
    el("label", { class: "field" }, "Age group ", age), el("label", { class: "field" }, "Native language(s) ", nat),
    el("label", { class: "field" }, "English proficiency ", prof),
    el("p", {}, "How familiar are you with these accents of English?"), el("table", { class: "fam" }, ...famRows),
    el("label", { class: "field" }, "Comments", comment),
    el("button", { class: "btn", onclick: () => {
      STATE.demographics = { age: age.value, native: nat.value, proficiency: prof.value,
        familiarity: Object.fromEntries(Object.entries(fam).map(([a, s]) => [a, s.value])), comment: comment.value };
      finish(); } }, "Finish"));
}

function result() {
  return { study: CONFIG.STUDY_ID, manifest_version: M.version, manifest_seed: M.seed, rater: R, pid: PID, dev: DEV,
    position: R % M.n_systems, started: STATE.started, finished: STATE.finished || new Date().toISOString(),
    user_agent: navigator.userAgent, setup: STATE.setup, demographics: STATE.demographics, responses: STATE.responses };
}
function download(name, text) {
  const a = el("a", { href: URL.createObjectURL(new Blob([text], { type: "application/json" })), download: name });
  document.body.append(a); a.click(); a.remove();
}

async function finish() {
  STATE.finished = new Date().toISOString(); save();
  const res = result();
  const code = "ACC-" + hashStr(`${R}|${PID}|${STATE.started}`).toString(36).toUpperCase();
  const status = el("p", { class: "status" }, CONFIG.SUBMIT_URL ? "Submitting…" : "");
  const file = `humaneval_${CONFIG.STUDY_ID}_r${R}${PID ? "_" + PID : ""}.json`;
  show(el("h1", {}, "Thank you!"), el("p", {}, "Your completion code:"), el("div", { class: "code" }, code),
    status, el("div", { class: "row" },
      el("button", { class: "btn secondary", onclick: () => download(file, JSON.stringify(res, null, 1)) }, "Download my answers (JSON)")),
    el("p", { class: "muted" }, "If the study organisers asked you to send the file, please attach the downloaded JSON."));
  if (!DEV) download(file, JSON.stringify(res, null, 1));
  if (CONFIG.SUBMIT_URL) {
    try {
      await fetch(CONFIG.SUBMIT_URL, { method: "POST", headers: { "Content-Type": "text/plain" }, body: JSON.stringify(res) });
      status.textContent = "Your answers were submitted."; status.className = "status ok";
    } catch (e) { status.textContent = "Automatic submission failed; please use the download button and send the file."; status.className = "status wait"; }
  }
}

// ── start ─────────────────────────────────────────────────────────────────────
function begin() {
  const rng = mulberry32(hashStr(`${M.seed}|${R}|${PID}`));
  STATE = { started: new Date().toISOString(), blocks: buildBlocks(M, R, rng), block: 0, trial: 0, responses: [], setup: null, demographics: null };
  save(); screenIntro();
}

fetch("manifest.json").then(r => r.json()).then(m => {
  M = m;
  document.getElementById("devbanner").hidden = !DEV;
  const rp = parseInt(params.get("r"), 10);
  R = Number.isFinite(rp) ? rp : (PID ? hashStr(PID) % 1000000 : Math.floor(Math.random() * 1000000));
  const prev = load();
  if (prev && !prev.finished && prev.responses.length > 0 && prev.responses.length < prev.blocks.reduce((n, x) => n + x.trials.length, 0)) {
    show(el("h1", {}, "Resume?"), el("p", {}, `You have an unfinished session (${prev.responses.length} recordings rated).`),
      el("div", { class: "row" },
        el("button", { class: "btn", onclick: () => { STATE = prev; STATE.trial = Math.min(STATE.trial, STATE.blocks[Math.min(STATE.block, STATE.blocks.length - 1)].trials.length - 1); nextBlockOrTrial(); } }, "Resume"),
        el("button", { class: "btn secondary", onclick: begin }, "Start over")));
  } else begin();
}).catch(e => show(el("h1", {}, "Could not load the test"), el("p", {}, String(e))));

function nextBlockOrTrial() { STATE.trial > 0 ? screenTrial() : nextBlock(); }
