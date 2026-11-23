// スマホのブラウザに出すページ(本体に内蔵。外部への通信はしない)
#pragma once
#include <Arduino.h>

// どちらのページでも使う見た目
#define STRIDER_PAGE_HEAD \
    "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">" \
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">" \
    "<style>" \
    ":root{--bg:#e9ecef;--card:#f8f9fa;--ink:#1d232c;--soft:#5b6573;--line:#cfd5dc;--red:#cf2e25;--red-ink:#fff}" \
    "@media(prefers-color-scheme:dark){:root{--bg:#12151a;--card:#1b2027;--ink:#e6e9ee;--soft:#98a2b1;--line:#2e3540;--red:#ff6257;--red-ink:#1f0705}}" \
    "*{box-sizing:border-box}body{margin:0;padding:20px 16px 32px;background:var(--bg);color:var(--ink);" \
    "font:16px/1.6 system-ui,-apple-system,'Hiragino Sans','Noto Sans JP',sans-serif}" \
    "main{max-width:26rem;margin:0 auto;display:flex;flex-direction:column;gap:14px}" \
    "h1{margin:0;font-size:22px;letter-spacing:.04em}h1 b{color:var(--red)}h2{margin:0;font-size:15px;color:var(--soft);font-weight:500}" \
    ".card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:12px}" \
    "button{font:inherit;font-weight:700;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--ink);padding:14px;min-height:52px}" \
    "button.go{background:var(--red);border-color:var(--red);color:var(--red-ink)}button[aria-pressed=true]{outline:3px solid var(--red);outline-offset:2px}" \
    "button:disabled{opacity:.5}label{display:flex;flex-direction:column;gap:4px;font-size:14px;color:var(--soft)}" \
    "input,select{font:inherit;padding:12px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);width:100%}" \
    "input[type=range]{padding:0;accent-color:var(--red)}input[type=checkbox]{width:auto}" \
    ".row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.wide{grid-column:1/-1}" \
    ".note{font-size:13px;color:var(--soft);margin:0}.state{font-variant-numeric:tabular-nums}" \
    "</style>"

// ---------------------------------------------------------------- 操作ページ(家の Wi-Fi につながっているとき)
static const char CONTROL_PAGE[] PROGMEM = STRIDER_PAGE_HEAD R"HTML(
<title>Stackchan Car</title>
<style>
/* 長押ししても文字が選択されたり、メニューが出たり、画面がスクロールしたりしないようにする */
button,.joy,.step{-webkit-user-select:none;user-select:none;-webkit-touch-callout:none;-webkit-tap-highlight-color:transparent;touch-action:manipulation}
main{min-height:calc(100dvh - 52px)}
.grow{flex:1}
.joy{position:relative;width:min(84vw,20rem);aspect-ratio:1;margin:0 auto;border-radius:50%;background:var(--card);border:2px solid var(--line);touch-action:none}
.joy::before,.joy::after{content:"";position:absolute;background:var(--line)}
.joy::before{left:50%;top:8%;bottom:8%;width:2px;margin-left:-1px}
.joy::after{top:50%;left:8%;right:8%;height:2px;margin-top:-1px}
.joy span{position:absolute;font-size:13px;color:var(--soft)}
.knob{position:absolute;left:50%;top:50%;width:34%;aspect-ratio:1;margin:-17% 0 0 -17%;border-radius:50%;background:var(--red);box-shadow:0 4px 12px rgba(0,0,0,.25);z-index:1;pointer-events:none}
.joy.held .knob{filter:brightness(1.1)}
.joy.busy{opacity:.45}
.seg{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.seg button[aria-pressed=true]{background:var(--ink);border-color:var(--ink);color:var(--bg);outline:0}
.step{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center}
.step output{min-width:4.5em;text-align:center;font-variant-numeric:tabular-nums;font-weight:700}
details summary{color:var(--soft);font-size:14px}
details .card{margin-top:10px}
</style></head><body><main>
<h1>STACKCHAN <b>CAR</b></h1>
<p class="note state" id="state">接続を確認しています…</p>
<p class="note state" id="pad"></p>
<p class="note state" id="atom"></p>
<div class="grow"></div>
<div class="joy" id="joy">
  <span style="top:2%;left:50%;transform:translateX(-50%)">前進</span>
  <span style="bottom:2%;left:50%;transform:translateX(-50%)">後退</span>
  <span style="left:3%;top:50%;transform:translateY(-50%)">左</span>
  <span style="right:3%;top:50%;transform:translateY(-50%)">右</span>
  <div class="knob" id="knob"></div>
</div>
<p class="note" style="text-align:center">指でつまみを動かします。上下で進む・下がる(倒すほど速い)、左右でかじを切ります。離すと止まります。</p>
<div class="grow"></div>
<details>
  <summary>設定</summary>
  <div class="card">
    <h2>車の種類(まっすぐ走るときの首の角度)</h2>
    <div class="seg" id="models">
      <button data-deg10="126" data-rev="0">car500</button>
      <button data-deg10="130" data-rev="0">レトロカー</button>
      <button data-deg10="150" data-rev="1">トライク</button>
    </div>
    <h2>まっすぐ走らないときの微調整</h2>
    <div class="step"><button id="trim-l">← 左へ</button><output id="neutral-out"></output><button id="trim-r">右へ →</button></div>
    <p class="note">まっすぐ進めたときに左へ曲がるなら「右へ」、右へ曲がるなら「左へ」を押します。本体に保存されます。</p>
    <label style="flex-direction:row;align-items:center;gap:8px;color:var(--ink)"><input type="checkbox" id="invert">前進と後退を入れ替える</label>
    <p class="note">つまみを上へ動かして後ろへ進む場合にチェックします。</p>
    <label style="flex-direction:row;align-items:center;gap:8px;color:var(--ink)"><input type="checkbox" id="smile">走っている間は笑顔にする</label>
    <button id="pad-forget">ゲームパッドを登録し直す</button>
    <button id="atom-forget">StopWatch を登録し直す</button>
    <p class="note">押してから 60 秒の間に、すぐそばに置いた StopWatch を、本体の画面で確かめて新しく登録します。BLE のゲームパッドに戻すときは、押したあと本体を再起動します。</p>
    <button id="forget">Wi-Fi の設定をやり直す</button>
  </div>
</details>
</main>
<script>
const $ = (id) => document.getElementById(id);
const call = (path) => fetch(path, { cache: "no-store" }).then((r) => r.json());
const PAD_STATES = ["探しています", "つないでいます", "つながっています", "使っていません(StopWatch を登録してあるため)"];
const ATOM_STATES = ["登録していません", "探しています", "つながっています"];
let neutral10 = 126, rev = 0, held = false, x = 0, y = 0, timer = null, busy = false, press = "", waiting = false;
function show(s) {
  $("invert").checked = !!s.invert;
  $("smile").checked = !!s.smile;
  neutral10 = Math.round(s.neutral * 10);
  rev = s.rev ? 1 : 0;
  $("neutral-out").textContent = s.neutral.toFixed(1) + "°";
  document.querySelectorAll("#models button").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.deg10) === neutral10 && Number(b.dataset.rev) === rev)));
  const moving = s.drive ? (s.drive > 0 ? "前進中" : "後退中") : "停止中";
  $("state").textContent = moving + " / かじ " + s.pitch.toFixed(1) + "° / 電池 " + s.volt.toFixed(2) + " V";
  $("pad").textContent = "ゲームパッド: " + (PAD_STATES[s.pad] || "-") + (s.padName ? "(" + s.padName + ")" : "");
  $("atom").textContent = "StopWatch: " + (ATOM_STATES[s.atom] || "-");
  busy = !!s.padBusy;
  $("joy").classList.toggle("busy", busy);
}
function lost() { $("state").textContent = "本体と通信できません。Wi-Fi を確認してください"; }

// つまみの位置 → -100〜100(右と前が正)。円の外へは出さない
function move(event) {
  const r = $("joy").getBoundingClientRect();
  const radius = r.width / 2 * 0.66;
  let dx = event.clientX - (r.left + r.width / 2), dy = event.clientY - (r.top + r.height / 2);
  const len = Math.hypot(dx, dy);
  if (len > radius) { dx *= radius / len; dy *= radius / len; }
  $("knob").style.transform = "translate(" + dx + "px," + dy + "px)";
  x = Math.round(dx / radius * 100);
  y = Math.round(-dy / radius * 100);
}
// 前の値の返事を待っている間は送らない(本体が遅いときに、値が溜まって遅れて届かないように)
function send() {
  if (waiting) return;
  waiting = true;
  call("/api/joy?x=" + x + "&y=" + y + "&s=" + press).then(show).catch(lost).finally(() => { waiting = false; });
}
// 触れている間は 0.1 秒ごとに送る。途切れると本体が自動で止まる
function start(event) {
  event.preventDefault();
  $("joy").setPointerCapture(event.pointerId);
  held = true;
  press = Math.random().toString(36).slice(2, 10);  // ひと押しごとの番号。離したあとに遅れて届いた値を、本体が捨てられるように
  waiting = false;
  $("joy").classList.add("held");
  move(event);
  send();
  clearInterval(timer);
  timer = setInterval(send, 100);
}
function stop() {
  if (!held) return;
  held = false;
  x = y = 0;
  clearInterval(timer);
  $("joy").classList.remove("held");
  $("knob").style.transform = "";
  call("/api/stop?s=" + press).then(show).catch(lost);
}
$("joy").addEventListener("pointerdown", start);
$("joy").addEventListener("pointermove", (event) => { if (held) move(event); });
["pointerup", "pointercancel", "lostpointercapture"].forEach((name) => $("joy").addEventListener(name, stop));
$("joy").addEventListener("contextmenu", (event) => event.preventDefault());
// 画面を切り替えた、ほかのアプリへ移った、などのときも止める
document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
window.addEventListener("blur", stop);
window.addEventListener("pagehide", stop);
document.addEventListener("selectstart", (event) => { if (event.target.closest("button,.joy,.step")) event.preventDefault(); });

const setNeutral = (deg10, query = "") => call("/api/neutral?deg10=" + deg10 + query).then(show).catch(lost);
$("models").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-deg10]");
  if (button) setNeutral(Number(button.dataset.deg10), "&rev=" + button.dataset.rev);
});
// 前輪でかじを取る車は首の角度が大きいほど右へ切れる。後輪でかじを取るトライクは逆
$("trim-l").onclick = () => setNeutral(neutral10 + (rev ? 5 : -5));
$("trim-r").onclick = () => setNeutral(neutral10 + (rev ? -5 : 5));
$("invert").onchange = () => call("/api/invert?on=" + ($("invert").checked ? 1 : 0)).then(show).catch(lost);
$("smile").onchange = () => call("/api/smile?on=" + ($("smile").checked ? 1 : 0)).then(show).catch(lost);
$("atom-forget").onclick = () => call("/api/atom/forget").then((s) => { show(s); $("atom").textContent = "StopWatch: 登録を消しました。60 秒の間に、すぐそばに置いた StopWatch を登録します"; }).catch(lost);
$("pad-forget").onclick = () => call("/api/pad/forget").then((s) => { show(s); $("pad").textContent = "ゲームパッドの登録を消しました。パッドをペアリング待ちにしてください"; }).catch(lost);
$("forget").onclick = () => {
  if ($("forget").dataset.armed) { call("/api/forget").catch(() => {}); $("state").textContent = "設定を消しました。本体の画面の QR を読んでください"; }
  else { $("forget").dataset.armed = "1"; $("forget").textContent = "もう一度押すと Wi-Fi の設定を消します"; }
};
// 操作していない間も、ときどき状態を読み直す(パッドの接続の表示のため)
setInterval(() => { if (!held && !document.hidden) call("/api/state").then(show).catch(lost); }, 2000);
call("/api/state").then(show).catch(lost);
</script></body></html>
)HTML";

// ---------------------------------------------------------------- 設定ページ(はじめて起動したとき / Wi-Fi につながらないとき)
static const char SETUP_PAGE[] PROGMEM = STRIDER_PAGE_HEAD R"HTML(
<title>Stackchan Strider の設定</title></head><body><main>
<h1>STACKCHAN <b>STRIDER</b></h1>
<form class="card" id="form">
  <h2>家の Wi-Fi につなぐ</h2>
  <label>Wi-Fi の名前
    <select id="ssid"><option value="">探しています…</option></select></label>
  <label>パスワード
    <input id="pass" type="password" autocomplete="off" autocapitalize="off"></label>
  <button class="go" id="save">この Wi-Fi につなぐ</button>
  <button type="button" id="rescan">もう一度探す</button>
  <p class="note" id="msg">2.4GHz の Wi-Fi だけ使えます。パスワードは本体の中にだけ保存されます。</p>
</form>
</main>
<script>
const $ = (id) => document.getElementById(id);
function scan() {
  $("ssid").innerHTML = '<option value="">探しています…</option>';
  fetch("/api/scan", { cache: "no-store" }).then((r) => r.json()).then((list) => {
    $("ssid").innerHTML = "";
    if (!list.length) { $("ssid").innerHTML = '<option value="">見つかりませんでした</option>'; return; }
    list.forEach((net) => { const o = document.createElement("option"); o.value = net.ssid; o.textContent = net.ssid + (net.lock ? "" : "(パスワードなし)"); $("ssid").appendChild(o); });
  }).catch(() => { $("msg").textContent = "Wi-Fi を探せませんでした。「もう一度探す」を押してください"; });
}
$("rescan").onclick = scan;
$("form").onsubmit = (event) => {
  event.preventDefault();
  if (!$("ssid").value) { $("msg").textContent = "Wi-Fi を選んでください"; return; }
  $("save").disabled = true;
  const body = new URLSearchParams({ ssid: $("ssid").value, pass: $("pass").value });
  fetch("/api/wifi", { method: "POST", body: body }).then(() => {
    $("msg").textContent = "保存しました。本体が再起動して Wi-Fi につなぎます。つながると本体の画面に操作ページの QR が出ます。出ない場合はパスワードを確かめて、もう一度この設定をしてください。";
  }).catch(() => { $("save").disabled = false; $("msg").textContent = "保存できませんでした。もう一度押してください"; });
};
scan();
</script></body></html>
)HTML";
