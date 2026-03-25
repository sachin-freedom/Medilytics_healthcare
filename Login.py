import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# Cursor spark canvas — rendered as a full-page overlay iframe.
# height must be > 0 for the iframe to exist; we give it 1px and position
# the canvas fixed relative to the PARENT window via JS.
# ─────────────────────────────────────────────────────────────────────────────
_SPARK_COMPONENT = """<!DOCTYPE html>
<html>
<head>
<style>
  html, body { margin:0; padding:0; background:transparent; overflow:hidden; }
</style>
</head>
<body>
<script>
(function(){
  /* ── Create canvas in the PARENT window ── */
  var pw   = window.parent;
  var pdoc = pw.document;

  /* Remove any old canvas from a previous Streamlit hot-reload */
  var old = pdoc.getElementById('medilytics-spark-canvas');
  if(old) old.remove();

  var canvas = pdoc.createElement('canvas');
  canvas.id  = 'medilytics-spark-canvas';
  Object.assign(canvas.style, {
    position:'fixed', inset:'0',
    width:'100vw', height:'100vh',
    pointerEvents:'none',
    zIndex:'99999',
    display:'block'
  });
  pdoc.body.appendChild(canvas);

  var ctx = canvas.getContext('2d');
  function resize(){
    canvas.width  = pw.innerWidth;
    canvas.height = pw.innerHeight;
  }
  resize();
  pw.addEventListener('resize', resize);

  /* ── Cursor dot ── */
  var old2 = pdoc.getElementById('medilytics-cursor-dot');
  if(old2) old2.remove();
  var dot = pdoc.createElement('div');
  dot.id  = 'medilytics-cursor-dot';
  Object.assign(dot.style, {
    position:'fixed',
    width:'9px', height:'9px',
    background:'#FFD700',
    borderRadius:'50%',
    pointerEvents:'none',
    zIndex:'999999',
    transform:'translate(-50%,-50%)',
    boxShadow:'0 0 8px #FFD700, 0 0 18px rgba(255,215,0,0.55)',
    transition:'transform .06s, opacity .1s',
    left:'-100px', top:'-100px'
  });
  pdoc.body.appendChild(dot);

  /* ── Inject cursor:none on the login page so our dot shows ── */
  var styleId = 'medilytics-cursor-override';
  var old3 = pdoc.getElementById(styleId);
  if(old3) old3.remove();
  var cs = pdoc.createElement('style');
  cs.id  = styleId;
  cs.textContent = '*, *::before, *::after { cursor: none !important; }';
  pdoc.head.appendChild(cs);

  /* ── Particle pool ── */
  var COLORS = ['#FFD700','#FFF59D','#FFECB3','#00E5FF','#ffffff','#FFD700'];
  var particles = [];
  var alive = true;

  function Spark(x, y){
    this.x  = x; this.y  = y;
    this.vx = (Math.random()-.5)*7;
    this.vy = (Math.random()-1.9)*6;
    this.life  = 1;
    this.decay = 0.028 + Math.random()*0.038;
    this.r     = 1.4 + Math.random()*2.6;
    this.color = COLORS[Math.floor(Math.random()*COLORS.length)];
    this.trail = [];
  }
  Spark.prototype.update = function(){
    this.trail.push({x:this.x, y:this.y});
    if(this.trail.length > 7) this.trail.shift();
    this.x  += this.vx;
    this.y  += this.vy;
    this.vy += 0.20;
    this.vx *= 0.97;
    this.life -= this.decay;
  };
  Spark.prototype.draw = function(){
    var t = this.trail;
    for(var i=0;i<t.length;i++){
      ctx.globalAlpha = (i/t.length)*this.life*0.45;
      ctx.beginPath();
      ctx.arc(t[i].x, t[i].y, this.r*0.55, 0, 6.28);
      ctx.fillStyle = this.color;
      ctx.fill();
    }
    ctx.globalAlpha = this.life*0.9;
    ctx.shadowBlur  = 10; ctx.shadowColor = this.color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.r, 0, 6.28);
    ctx.fillStyle = this.color;
    ctx.fill();
    ctx.shadowBlur = 0; ctx.globalAlpha = 1;
  };

  function burst(x,y,n){
    if(!alive) return;
    for(var i=0;i<n;i++) particles.push(new Spark(x,y));
  }

  /* ── Cleanup: remove all spark artefacts from the parent DOM ── */
  function cleanup(){
    alive = false;
    /* Remove canvas */
    var c = pdoc.getElementById('medilytics-spark-canvas');
    if(c) c.remove();
    /* Remove cursor dot */
    var d = pdoc.getElementById('medilytics-cursor-dot');
    if(d) d.remove();
    /* Remove cursor:none override so dashboard gets normal cursor back */
    var s = pdoc.getElementById('medilytics-cursor-override');
    if(s) s.remove();
  }

  /* ── Self-destruct when THIS iframe is removed from the parent DOM ── */
  /* Streamlit removes the iframe component when it stops rendering (i.e. after login) */
  var observer = new MutationObserver(function(){
    /* If our iframe's src iframe element is no longer in the parent document,
       it means Streamlit has navigated away from the login page → clean up. */
    if(!pdoc.body.contains(window.frameElement)){
      observer.disconnect();
      cleanup();
    }
  });
  if(pdoc.body){
    observer.observe(pdoc.body, {childList:true, subtree:true});
  }

  /* ── Mouse tracking ── */
  var lx=-999,ly=-999;
  function onMouseMove(e){
    if(!alive) return;
    var dx=e.clientX-lx, dy=e.clientY-ly;
    var sp=Math.sqrt(dx*dx+dy*dy);
    if(sp>3) burst(e.clientX, e.clientY, Math.min(Math.floor(sp*0.55),10));
    lx=e.clientX; ly=e.clientY;
    dot.style.left = e.clientX+'px';
    dot.style.top  = e.clientY+'px';
  }
  function onClick(e){ burst(e.clientX,e.clientY,22); }
  function onLeave(){ if(dot) dot.style.opacity='0'; }
  function onEnter(){ if(dot) dot.style.opacity='1'; }
  function onDown(){
    if(!dot) return;
    dot.style.transform='translate(-50%,-50%) scale(2.2)';
    dot.style.opacity='0.55';
  }
  function onUp(){
    if(!dot) return;
    dot.style.transform='translate(-50%,-50%) scale(1)';
    dot.style.opacity='1';
  }

  pw.addEventListener('mousemove',  onMouseMove);
  pw.addEventListener('click',      onClick);
  pw.addEventListener('mouseleave', onLeave);
  pw.addEventListener('mouseenter', onEnter);
  pw.addEventListener('mousedown',  onDown);
  pw.addEventListener('mouseup',    onUp);

  /* ── Render loop ── */
  function loop(){
    if(!alive){
      /* Final clear so no ghost particles remain */
      ctx.clearRect(0,0,canvas.width,canvas.height);
      return;
    }
    ctx.clearRect(0,0,canvas.width,canvas.height);
    for(var i=particles.length-1;i>=0;i--){
      particles[i].update();
      particles[i].draw();
      if(particles[i].life<=0) particles.splice(i,1);
    }
    requestAnimationFrame(loop);
  }
  loop();
})();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# CSS for the login page background, ECG, blobs, and intro overlay
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Hide Streamlit chrome ── */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu, footer { display:none!important }

/* ── Full-page dark background ── */
html, body, .stApp,
[data-testid="stAppViewContainer"] {
    background: #020817 !important;
    min-height: 100vh !important;
}

/* ── Animated grid ── */
.stApp::before {
    content:'';
    position:fixed; inset:0;
    background-image:
        linear-gradient(rgba(255,215,0,0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,215,0,0.045) 1px, transparent 1px);
    background-size:52px 52px;
    animation:gridMove 22s linear infinite;
    pointer-events:none; z-index:0;
}
@keyframes gridMove {
    0%   { background-position:0 0,0 0; }
    100% { background-position:52px 52px,52px 52px; }
}

/* ── Gold blob top-left ── */
.stApp::after {
    content:'';
    position:fixed; width:600px; height:600px;
    background:radial-gradient(circle, rgba(255,215,0,0.13) 0%, transparent 65%);
    top:-180px; left:-160px;
    border-radius:50%; filter:blur(80px);
    pointer-events:none; z-index:0;
    animation:blobA 11s ease-in-out infinite alternate;
}
@keyframes blobA {
    0%   { transform:scale(1) translate(0,0); }
    100% { transform:scale(1.2) translate(28px,20px); }
}

/* ── Centering ── */
[data-testid="stMainBlockContainer"] {
    padding:0 !important;
    max-width:100% !important;
    min-height:100vh !important;
    display:flex !important;
    flex-direction:column !important;
    align-items:center !important;
    justify-content:center !important;
}
[data-testid="stMainBlockContainer"] > div[data-testid="block-container"],
[data-testid="block-container"] {
    padding:0 1rem !important;
    width:100% !important;
    max-width:100% !important;
    display:flex !important;
    flex-direction:column !important;
    align-items:center !important;
    justify-content:center !important;
    flex:1 !important;
    min-height:100vh !important;
}
[data-testid="stHorizontalBlock"] {
    width:100% !important;
    justify-content:center !important;
    align-items:center !important;
}

/* ── Ambient blobs ── */
.blob-cyan {
    position:fixed; width:500px; height:500px;
    background:radial-gradient(circle, rgba(0,229,255,0.10) 0%, transparent 65%);
    bottom:-120px; right:-120px;
    border-radius:50%; filter:blur(80px);
    pointer-events:none; z-index:0;
    animation:blobB 13s ease-in-out infinite alternate;
}
@keyframes blobB {
    0%   { transform:scale(1) translate(0,0); }
    100% { transform:scale(1.15) translate(-22px,-28px); }
}
.blob-purple {
    position:fixed; width:360px; height:360px;
    background:radial-gradient(circle, rgba(167,139,250,0.09) 0%, transparent 65%);
    top:38%; left:60%;
    border-radius:50%; filter:blur(80px);
    pointer-events:none; z-index:0;
    animation:blobC 9s ease-in-out infinite alternate;
}
@keyframes blobC {
    0%   { transform:scale(1) translate(0,0); }
    100% { transform:scale(1.18) translate(-15px,20px); }
}

/* ── ECG lines ── */
.ecg-wrap {
    position:fixed; inset:0;
    pointer-events:none; z-index:1; overflow:hidden;
}
.ecg-line {
    position:absolute; left:0; right:0; height:160px;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='160' viewBox='0 0 600 160'%3E%3Cpolyline fill='none' stroke='%23FFD700' stroke-width='1.5' opacity='0.6' stroke-linecap='round' stroke-linejoin='round' points='0,80 55,80 75,80 88,72 100,80 104,72 107,20 109,140 111,65 120,80 175,80 195,80 208,72 220,80 224,72 227,20 229,140 231,65 240,80 295,80 315,80 328,72 340,80 344,72 347,20 349,140 351,65 360,80 415,80 435,80 448,72 460,80 464,72 467,20 469,140 471,65 480,80 535,80 555,80 568,72 580,80 584,72 587,20 589,140 591,65 600,80'/%3E%3C/svg%3E");
    background-repeat:repeat-x; background-size:600px 160px;
    animation:ecgScroll linear infinite;
}
.ecg-line:nth-child(1){ top:10%; opacity:.55; animation-duration:8s; }
.ecg-line:nth-child(2){ top:34%; opacity:.32; animation-duration:11s; }
.ecg-line:nth-child(3){ top:58%; opacity:.22; animation-duration:14s; }
.ecg-line:nth-child(4){ top:80%; opacity:.15; animation-duration:17s; }
@keyframes ecgScroll {
    0%   { background-position:0 0; }
    100% { background-position:-600px 0; }
}

/* ── INTRO OVERLAY ── */
#intro-overlay {
    position:fixed; inset:0; z-index:5000;
    background:#020817;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    animation:introFadeOut 0.7s ease 3.0s forwards;
}
@keyframes introFadeOut {
    0%   { opacity:1; pointer-events:all; }
    100% { opacity:0; pointer-events:none; }
}
.intro-scanline {
    position:absolute; left:0; right:0; height:3px;
    background:linear-gradient(90deg, transparent, rgba(255,215,0,0.7), transparent);
    top:0;
    animation:scanDown 1.6s ease 0.4s forwards;
}
@keyframes scanDown {
    0%   { top:0; opacity:1; }
    100% { top:100%; opacity:0; }
}
.intro-logo {
    font-family:'Rajdhani',sans-serif;
    font-size:64px; font-weight:700;
    color:transparent;
    letter-spacing:16px; text-transform:uppercase;
    background:linear-gradient(90deg,#FFD700 0%,#fff 50%,#FFD700 100%);
    background-size:200% auto;
    -webkit-background-clip:text; background-clip:text;
    animation:logoShimmer 1.6s ease 0.3s both, logoPulse 0.4s ease 2.5s forwards;
}
@keyframes logoShimmer {
    0%   { opacity:0; background-position:200% center; letter-spacing:30px; }
    60%  { opacity:1; }
    100% { opacity:1; background-position:-200% center; letter-spacing:16px; }
}
@keyframes logoPulse {
    0%   { transform:scale(1); opacity:1; }
    50%  { transform:scale(1.05); }
    100% { transform:scale(0.9); opacity:0; }
}
.intro-line {
    width:0; height:2px; margin-top:14px;
    background:linear-gradient(90deg,transparent,#FFD700,transparent);
    animation:lineGrow 0.9s ease 0.8s forwards;
}
@keyframes lineGrow { to { width:320px; } }
.intro-tag {
    font-family:'JetBrains Mono',monospace;
    font-size:11px; letter-spacing:5px;
    color:#475569; text-transform:uppercase;
    margin-top:14px; opacity:0;
    animation:tagFade 0.7s ease 1.3s forwards;
}
@keyframes tagFade { to { opacity:1; } }

/* ── Login card ── */
.lcard {
    background:rgba(10,22,42,0.95);
    border:1px solid rgba(255,215,0,0.25);
    border-radius:20px;
    padding:38px 44px 32px;
    width:100%;
    backdrop-filter:blur(32px); -webkit-backdrop-filter:blur(32px);
    box-shadow:0 0 0 1px rgba(255,215,0,0.08), 0 28px 70px rgba(0,0,0,0.75);
    animation:cardIn 0.7s cubic-bezier(.22,1,.36,1) 3.1s both,
              cardGlow 4s ease-in-out 4.2s infinite alternate;
    position:relative; z-index:10;
}
@keyframes cardIn {
    from { opacity:0; transform:translateY(26px) scale(.97); }
    to   { opacity:1; transform:translateY(0) scale(1); }
}
@keyframes cardGlow {
    0%   { box-shadow:0 0 0 1px rgba(255,215,0,.09),0 28px 70px rgba(0,0,0,.75); }
    100% { box-shadow:0 0 0 1px rgba(255,215,0,.32),0 28px 70px rgba(0,0,0,.75),
                      0 0 55px rgba(255,215,0,.10); }
}
.lbrand {
    font-family:'Rajdhani',sans-serif;
    font-size:38px; font-weight:700;
    color:#FFD700; letter-spacing:6px;
    text-transform:uppercase; text-align:center;
    text-shadow:0 0 30px rgba(255,215,0,0.45);
    margin-bottom:4px;
    animation:fadeD 0.6s ease 3.3s both;
}
@keyframes fadeD {
    from { opacity:0; transform:translateY(-8px); }
    to   { opacity:1; transform:translateY(0); }
}
.laccent {
    height:2px;
    background:linear-gradient(90deg,transparent,#FFD700 40%,transparent);
    border-radius:2px; margin-bottom:16px;
    animation:lineX 0.9s ease 3.5s both; transform-origin:center;
}
@keyframes lineX {
    from { transform:scaleX(0); opacity:0; }
    to   { transform:scaleX(1); opacity:1; }
}
.ltag {
    font-family:'JetBrains Mono',monospace;
    font-size:10px; letter-spacing:3px; color:#475569;
    text-transform:uppercase; text-align:center; margin-bottom:6px;
    animation:fadeD 0.6s ease 3.6s both;
}

/* ── Input fields ── */
[data-testid="column"]:nth-child(2) .stTextInput > label {
    font-family:'Rajdhani',sans-serif !important;
    font-size:11px !important; font-weight:700 !important;
    letter-spacing:1.3px !important; text-transform:uppercase !important;
    color:#64748B !important;
}
[data-testid="column"]:nth-child(2) .stTextInput input {
    background:rgba(255,255,255,.04) !important;
    border:1px solid rgba(255,255,255,.10) !important;
    border-radius:8px !important; color:#FFFFFF !important;
    font-family:'Inter',sans-serif !important; font-size:14px !important;
    transition:border-color .2s, box-shadow .2s !important;
}
[data-testid="column"]:nth-child(2) .stTextInput input:focus {
    border-color:#FFD700 !important;
    box-shadow:0 0 0 3px rgba(255,215,0,.13) !important;
    background:rgba(255,215,0,.03) !important;
}
[data-testid="column"]:nth-child(2) .stTextInput input::placeholder {
    color:#8BA3BF !important;
}

/* ── Sign-in button ── */
[data-testid="column"]:nth-child(2) .stButton > button {
    background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
    border:none !important; border-radius:8px !important; color:#fff !important;
    font-family:'Rajdhani',sans-serif !important; font-size:15px !important;
    font-weight:700 !important; letter-spacing:2px !important;
    text-transform:uppercase !important; height:48px !important; width:100% !important;
    box-shadow:0 4px 22px rgba(99,102,241,.42) !important;
    transition:transform .18s, box-shadow .18s !important;
    margin-top:6px !important;
}
[data-testid="column"]:nth-child(2) .stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 30px rgba(99,102,241,.58) !important;
}

.lfooter {
    text-align:center; margin-top:18px;
    font-family:'JetBrains Mono',monospace;
    font-size:9px; letter-spacing:2px; color:#1E293B;
    text-transform:uppercase; position:relative; z-index:10;
    animation:fadeD 0.6s ease 3.9s both;
}

/* Staggered fade-in for widgets */
[data-testid="column"]:nth-child(2) .stTextInput {
    animation:fadeD 0.5s ease 3.5s both;
}
[data-testid="column"]:nth-child(2) .stButton {
    animation:fadeD 0.5s ease 3.7s both;
}
</style>

<div class="blob-cyan"></div>
<div class="blob-purple"></div>
<div class="ecg-wrap">
  <div class="ecg-line"></div>
  <div class="ecg-line"></div>
  <div class="ecg-line"></div>
  <div class="ecg-line"></div>
</div>

<!-- Animated intro overlay -->
<div id="intro-overlay">
  <div class="intro-scanline"></div>
  <div class="intro-logo">MEDILYTICS</div>
  <div class="intro-line"></div>
  <div class="intro-tag">Healthcare Revenue Intelligence</div>
</div>
"""


def show_login():
    # ── Init session state ────────────────────────────────────────────────────
    for k, v in [("logged_in", False), ("role", None), ("username", None),
                 ("department", None), ("page", "executive"), ("filters", {})]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Load users ────────────────────────────────────────────────────────────
    users = pd.read_csv("data/users.csv")
    users = users.loc[:, ~users.columns.str.startswith("Unnamed")]
    users.columns = [c.strip() for c in users.columns]

    # ── Inject page CSS + intro overlay ──────────────────────────────────────
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Inject cursor spark canvas into PARENT window via iframe ─────────────
    # height=1 ensures the iframe is mounted; JS then works in parent DOM
    components.html(_SPARK_COMPONENT, height=1, scrolling=False)

    # ── Layout ────────────────────────────────────────────────────────────────
    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        st.markdown("""
        <div class="lcard">
          <div class="lbrand">MEDILYTICS</div>
          <div class="laccent"></div>
          <div class="ltag">Healthcare Revenue Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input(
            "Username", placeholder="Enter your username", key="lg_user")
        password = st.text_input(
            "Password", placeholder="Enter your password",
            type="password", key="lg_pass")

        if st.button("Sign In", use_container_width=True, key="lg_btn"):
            _auth(username.strip(), password.strip(), users)

        st.markdown(
            '<div class="lfooter">Medilytics v2.0 &nbsp;&middot;&nbsp; '
            'Secure Healthcare Analytics</div>',
            unsafe_allow_html=True
        )


def _auth(username: str, password: str, users: pd.DataFrame):
    """
    Authenticate against data/users.csv.
    Required columns: username, password, role, department
    """
    if not username or not password:
        st.error("Please enter both username and password.")
        return

    match = users[
        (users["username"] == username) &
        (users["password"].astype(str) == password)
    ]

    if not match.empty:
        row = match.iloc[0]
        st.session_state.logged_in  = True
        st.session_state.username   = username
        st.session_state.role       = row["role"]
        st.session_state.department = row["department"]
        st.rerun()
    else:
        st.error("Invalid credentials — please check username and password.")
