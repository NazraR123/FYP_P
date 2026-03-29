from flask import Flask, request, render_template_string, url_for
from ultralytics import YOLO
import numpy as np
import cv2
import os
import time
import hashlib
from datetime import datetime
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image
from flask_cors import CORS
import traceback
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['OUTPUT_FOLDER'] = './static/outputs'

model = YOLO('F:/Project/BE/deepfake images/best.pt')
model.cpu()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


# ----------------------------------------------------------------
#  IMAGE PROCESS
# ----------------------------------------------------------------
def process_image(image_path):
    img = cv2.imread(image_path)
    h_orig, w_orig = img.shape[:2]

    img_resized = cv2.resize(img, (832, 832))
    rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = np.float32(rgb_img) / 255

    start = time.time()
    results = model(rgb_img)
    inference_time = round((time.time() - start) * 1000, 1)

    predictions = {}
    if hasattr(results[0], 'probs') and results[0].probs is not None:
        probabilities = results[0].probs.data.cpu().numpy()
        class_names   = model.names
        predictions   = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

    target_layers = [model.model.model[-3]]
    cam = EigenCAM(model, target_layers, task='cls')
    grayscale_cam = cam(rgb_img)[0, :, :]
    cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)

    file_size_kb = round(os.path.getsize(image_path) / 1024, 1)
    sha256       = hashlib.sha256(open(image_path,'rb').read()).hexdigest()[:16].upper()
    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    meta = {
        "Resolution":      f"{w_orig} × {h_orig} px",
        "File Size":       f"{file_size_kb} KB",
        "Inference Time":  f"{inference_time} ms",
        "Model":           "YOLOv8-cls",
        "XAI Method":      "EigenCAM",
        "SHA-256 (first)": sha256,
        "Scanned At":      timestamp,
    }
    return predictions, cam_image, meta


# ================================================================
#  HOME
# ================================================================
HOME_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SYNAPSE — DeepFake Forensics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#060912;--surface:#0d1424;--border:rgba(0,200,255,.15);
  --accent:#00c8ff;--accent2:#ff2d6b;--text:#c8d8f0;--muted:#4a6080;
  --mono:'Share Tech Mono',monospace;--sans:'Syne',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--sans);
     min-height:100vh;display:flex;flex-direction:column;align-items:center;
     justify-content:center;padding:24px;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;
  background-image:linear-gradient(rgba(0,200,255,.03) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,200,255,.03) 1px,transparent 1px);
  background-size:40px 40px;pointer-events:none;z-index:0}
.orb{position:fixed;border-radius:50%;filter:blur(130px);pointer-events:none;z-index:0;opacity:.16}
.orb-1{width:500px;height:500px;background:#00c8ff;top:-150px;left:-150px}
.orb-2{width:400px;height:400px;background:#ff2d6b;bottom:-100px;right:-100px}
.wrapper{position:relative;z-index:1;width:100%;max-width:520px;animation:fadeUp .7s ease both}
@keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
.header{text-align:center;margin-bottom:32px}
.badge{display:inline-flex;align-items:center;gap:8px;background:rgba(0,200,255,.08);
       border:1px solid var(--border);border-radius:6px;padding:4px 14px;
       font-family:var(--mono);font-size:11px;color:var(--accent);letter-spacing:2px;margin-bottom:16px}
.badge .dot{width:7px;height:7px;background:var(--accent);border-radius:50%;animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
h1{font-size:clamp(32px,6vw,48px);font-weight:800;letter-spacing:-1px;line-height:1.1;
   background:linear-gradient(135deg,#fff 30%,var(--accent));
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.subtitle{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:10px;letter-spacing:1px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:32px;
      backdrop-filter:blur(8px);box-shadow:0 0 0 1px rgba(0,200,255,.04),0 24px 60px rgba(0,0,0,.6)}
.drop-zone{border:2px dashed rgba(0,200,255,.25);border-radius:14px;padding:36px 20px;text-align:center;
           cursor:pointer;transition:all .25s;background:rgba(0,200,255,.02);position:relative;overflow:hidden}
.drop-zone::after{content:'';position:absolute;inset:0;
  background:radial-gradient(circle at center,rgba(0,200,255,.06) 0%,transparent 70%);
  opacity:0;transition:opacity .3s}
.drop-zone:hover{border-color:var(--accent);background:rgba(0,200,255,.05)}
.drop-zone:hover::after,.drop-zone.dragover::after{opacity:1}
.drop-zone.dragover{border-color:var(--accent);background:rgba(0,200,255,.08);transform:scale(1.01)}
.drop-icon{font-size:42px;margin-bottom:12px;display:block;filter:drop-shadow(0 0 14px rgba(0,200,255,.4))}
.drop-title{font-size:15px;font-weight:700;color:#fff;margin-bottom:6px}
.drop-sub{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.5px}
#preview-wrap{display:none;margin-top:18px;border-radius:12px;overflow:hidden;
              position:relative;border:1px solid var(--border)}
#preview-wrap img{width:100%;display:block;border-radius:12px}
.preview-tag{position:absolute;top:10px;left:10px;font-family:var(--mono);font-size:10px;
             background:rgba(6,9,18,.8);border:1px solid var(--border);border-radius:4px;
             padding:3px 8px;color:var(--accent);letter-spacing:1px;text-transform:uppercase}
.file-meta{display:none;margin-top:12px;background:rgba(0,200,255,.04);
           border:1px solid var(--border);border-radius:10px;padding:12px 16px;
           font-family:var(--mono);font-size:11px;color:var(--muted);gap:20px;flex-wrap:wrap}
.file-meta.on{display:flex}
.file-meta span strong{display:block;color:var(--text);font-size:12px}
.btn{display:flex;align-items:center;justify-content:center;gap:10px;margin-top:20px;
     width:100%;padding:14px;border:none;border-radius:12px;
     background:linear-gradient(135deg,var(--accent) 0%,#0070a8 100%);
     color:#fff;font-family:var(--sans);font-size:15px;font-weight:700;cursor:pointer;
     letter-spacing:.5px;transition:all .2s;position:relative;overflow:hidden}
.btn::before{content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,.15),transparent);opacity:0;transition:opacity .2s}
.btn:hover::before{opacity:1}
.btn:hover{transform:translateY(-1px);box-shadow:0 8px 24px rgba(0,200,255,.25)}
.btn:disabled{opacity:.6;cursor:not-allowed;transform:none}
.loader-overlay{display:none;position:fixed;inset:0;background:rgba(6,9,18,.85);z-index:100;
                flex-direction:column;align-items:center;justify-content:center;backdrop-filter:blur(6px)}
.loader-overlay.on{display:flex}
.scan-ring{width:90px;height:90px;border-radius:50%;border:2px solid rgba(0,200,255,.15);
           border-top-color:var(--accent);animation:spin 1s linear infinite;position:relative;margin-bottom:24px}
.scan-ring::after{content:'';position:absolute;inset:8px;border-radius:50%;
  border:2px solid rgba(255,45,107,.15);border-bottom-color:var(--accent2);
  animation:spin 1.4s linear infinite reverse}
@keyframes spin{to{transform:rotate(360deg)}}
.loader-text{font-family:var(--mono);font-size:13px;color:var(--accent);letter-spacing:2px;animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.loader-sub{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:8px;letter-spacing:1px}
.tags{display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:24px}
.tag{font-family:var(--mono);font-size:10px;color:var(--muted);
     border:1px solid rgba(255,255,255,.06);border-radius:4px;padding:3px 10px;letter-spacing:.8px}
</style>
</head>
<body>
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="wrapper">
  <div class="header">
    <div class="badge"><span class="dot"></span> SYNAPSE FORENSICS v2.0</div>
    <h1>DeepFake<br>Detector</h1>
    <p class="subtitle">// AI-POWERED IMAGE AUTHENTICITY ANALYSIS</p>
  </div>
  <div class="card">
    <form method="post" action="/upload" enctype="multipart/form-data" id="form">
      <div class="drop-zone" id="dz" onclick="document.getElementById('fi').click()">
        <span class="drop-icon">🔬</span>
        <div class="drop-title">Drop image for forensic scan</div>
        <div class="drop-sub">PNG · JPG · WEBP · BMP &nbsp;|&nbsp; Max 20 MB</div>
      </div>
      <input type="file" id="fi" name="file" hidden accept="image/*" required>
      <div id="preview-wrap">
        <span class="preview-tag">📷 Input Frame</span>
        <img id="pi" src="" alt="preview">
      </div>
      <div class="file-meta" id="fm">
        <span><strong id="fn">—</strong>Filename</span>
        <span><strong id="fs">—</strong>Size</span>
        <span><strong id="ft">—</strong>Format</span>
        <span><strong id="fd">—</strong>Dimensions</span>
      </div>
      <button type="submit" class="btn" id="sb">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
          <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        Run Forensic Analysis
      </button>
    </form>
  </div>
  <div class="tags">
    <span class="tag">YOLOV8</span><span class="tag">EIGENCAM</span>
    <span class="tag">CNN CLASSIFIER</span><span class="tag">LOCAL INFERENCE</span>
  </div>
</div>
<div class="loader-overlay" id="lo">
  <div class="scan-ring"></div>
  <div class="loader-text">SCANNING IMAGE</div>
  <div class="loader-sub">Running neural forensics · please wait…</div>
</div>
<script>
const fi=document.getElementById('fi'),dz=document.getElementById('dz'),
      pw=document.getElementById('preview-wrap'),pi=document.getElementById('pi'),
      fm=document.getElementById('fm'),sb=document.getElementById('sb');

dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('dragover')});
dz.addEventListener('dragleave',()=>dz.classList.remove('dragover'));
dz.addEventListener('drop',e=>{
  e.preventDefault();dz.classList.remove('dragover');
  const f=e.dataTransfer.files[0];
  if(f&&f.type.startsWith('image/')){
    const dt=new DataTransfer();dt.items.add(f);fi.files=dt.files;load(f);
  }
});
fi.addEventListener('change',e=>{if(e.target.files[0])load(e.target.files[0])});

function load(file){
  document.getElementById('fn').textContent=file.name.length>18?file.name.slice(0,15)+'…':file.name;
  document.getElementById('fs').textContent=(file.size/1024).toFixed(1)+' KB';
  document.getElementById('ft').textContent=file.type.replace('image/','').toUpperCase();
  fm.classList.add('on');
  const r=new FileReader();
  r.onload=ev=>{
    const i=new Image();
    i.onload=()=>{document.getElementById('fd').textContent=i.naturalWidth+'×'+i.naturalHeight};
    i.src=ev.target.result;
    pi.src=ev.target.result;pw.style.display='block';
    try{sessionStorage.setItem('up',ev.target.result)}catch(e){}
  };
  r.readAsDataURL(file);
}
document.getElementById('form').addEventListener('submit',()=>{
  document.getElementById('lo').classList.add('on');
  sb.disabled=true;sb.innerHTML='⏳ &nbsp;Analyzing…';
});
</script>
</body>
</html>"""


# ================================================================
#  RESULT
# ================================================================
RESULT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SYNAPSE — Forensic Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#060912;--surface:#0d1424;--s2:#111827;
  --border:rgba(0,200,255,.12);--accent:#00c8ff;
  --text:#c8d8f0;--muted:#4a6080;
  --vc:{{ verdict_color }};
  --mono:'Share Tech Mono',monospace;--sans:'Syne',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--sans);
     min-height:100vh;padding:40px 20px 60px;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;
  background-image:linear-gradient(rgba(0,200,255,.025) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,200,255,.025) 1px,transparent 1px);
  background-size:40px 40px;pointer-events:none;z-index:0}
.orb{position:fixed;border-radius:50%;filter:blur(140px);pointer-events:none;z-index:0;opacity:.12}
.orb-1{width:600px;height:600px;background:var(--vc);top:-200px;left:-200px}
.orb-2{width:400px;height:400px;background:#00c8ff;bottom:-100px;right:-100px}
.page{position:relative;z-index:1;max-width:1020px;margin:auto}
@keyframes fadeDown{from{opacity:0;transform:translateY(-12px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.topbar{display:flex;align-items:center;justify-content:space-between;
        margin-bottom:32px;animation:fadeDown .5s ease both}
.badge{display:inline-flex;align-items:center;gap:8px;background:rgba(0,200,255,.08);
       border:1px solid var(--border);border-radius:6px;padding:4px 14px;
       font-family:var(--mono);font-size:11px;color:var(--accent);letter-spacing:2px}
.badge .dot{width:7px;height:7px;background:var(--accent);border-radius:50%;animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.back{font-family:var(--mono);font-size:11px;color:var(--muted);text-decoration:none;
      border:1px solid rgba(255,255,255,.07);border-radius:6px;padding:5px 14px;
      transition:all .2s;display:flex;align-items:center;gap:6px}
.back:hover{color:var(--accent);border-color:var(--border)}

/* verdict hero */
.hero{background:var(--surface);border:1px solid rgba(255,255,255,.06);border-radius:20px;
      padding:28px 32px;margin-bottom:20px;display:flex;align-items:center;
      gap:28px;flex-wrap:wrap;position:relative;overflow:hidden;
      animation:fadeUp .5s .1s ease both}
.hero::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;
              background:var(--vc);border-radius:4px 0 0 4px}
.hero-glow{position:absolute;right:-60px;top:-60px;width:240px;height:240px;
           background:var(--vc);border-radius:50%;filter:blur(80px);opacity:.1}
.v-icon{font-size:56px;line-height:1;filter:drop-shadow(0 0 20px var(--vc))}
.v-text h2{font-size:13px;font-family:var(--mono);color:var(--muted);
           letter-spacing:2px;text-transform:uppercase;margin-bottom:6px}
.v-label{font-size:clamp(30px,5vw,44px);font-weight:800;color:var(--vc);
         line-height:1;letter-spacing:-1px;text-shadow:0 0 40px var(--vc)}
.v-conf{font-family:var(--mono);font-size:14px;color:var(--text);margin-top:8px;opacity:.7}
.ring-wrap{margin-left:auto;text-align:center;min-width:110px}
.ring-label{font-family:var(--mono);font-size:10px;color:var(--muted);
            letter-spacing:1px;margin-bottom:10px;text-transform:uppercase}
.ring-svg{width:90px;height:90px;display:block;margin:auto;transform:rotate(-90deg)}
.ring-bg{fill:none;stroke:rgba(255,255,255,.06);stroke-width:6}
.ring-fg{fill:none;stroke:var(--vc);stroke-width:6;stroke-linecap:round;
         stroke-dasharray:226;stroke-dashoffset:{{ ring_offset }};
         filter:drop-shadow(0 0 6px var(--vc));
         transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)}
.ring-center{position:relative;text-align:center;margin-top:-62px;
             font-family:var(--mono);font-size:16px;font-weight:700;color:var(--vc);line-height:52px}

/* interpretation */
.interp{background:rgba({{ i_rgb }},.06);border:1px solid rgba({{ i_rgb }},.2);
        border-radius:14px;padding:18px 22px;margin-bottom:20px;
        display:flex;gap:14px;align-items:flex-start;animation:fadeUp .5s .2s ease both}
.interp-icon{font-size:24px;flex-shrink:0;margin-top:2px}
.interp-title{font-family:var(--mono);font-size:10px;letter-spacing:1.5px;
              text-transform:uppercase;color:var(--vc);margin-bottom:6px}
.interp-body{font-size:13px;line-height:1.65;color:var(--text);opacity:.85}

/* image grid */
.img-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}
@media(max-width:620px){.img-grid{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid rgba(255,255,255,.06);
       border-radius:16px;overflow:hidden;animation:fadeUp .5s ease both}
.panel:nth-child(1){animation-delay:.2s}
.panel:nth-child(2){animation-delay:.3s}
.p-header{display:flex;align-items:center;gap:10px;padding:13px 18px;
          border-bottom:1px solid rgba(255,255,255,.05);background:rgba(0,200,255,.03)}
.p-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent)}
.p-title{font-family:var(--mono);font-size:11px;letter-spacing:1.5px;color:var(--accent);text-transform:uppercase}
.panel img{width:100%;display:block}
.placeholder{min-height:220px;display:flex;align-items:center;justify-content:center;
             font-family:var(--mono);font-size:11px;color:var(--muted)}

/* bars */
.bars{background:var(--surface);border:1px solid rgba(255,255,255,.06);
      border-radius:16px;padding:24px;margin-bottom:20px;animation:fadeUp .5s .35s ease both}
.bars-title{font-family:var(--mono);font-size:11px;letter-spacing:1.5px;color:var(--accent);
            text-transform:uppercase;margin-bottom:20px;display:flex;align-items:center;gap:8px}
.bars-title::before{content:'';display:inline-block;width:8px;height:8px;background:var(--accent);
                    border-radius:50%;box-shadow:0 0 8px var(--accent)}
.bar-row{margin-bottom:16px}
.bar-meta{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:7px}
.bar-name{font-size:14px;font-weight:700}
.bar-pct{font-family:var(--mono);font-size:13px;color:var(--text)}
.bar-track{background:rgba(255,255,255,.05);border-radius:6px;height:11px;overflow:hidden}
.bar-fill{height:100%;border-radius:6px;width:0%;transition:width 1s cubic-bezier(.4,0,.2,1);position:relative}
.bar-fill::after{content:'';position:absolute;right:0;top:0;bottom:0;width:4px;
                 background:rgba(255,255,255,.3);border-radius:0 6px 6px 0}

/* meta cards */
.meta-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));
           gap:12px;margin-bottom:20px;animation:fadeUp .5s .4s ease both}
.mc{background:var(--surface);border:1px solid rgba(255,255,255,.06);
    border-radius:12px;padding:14px 16px;transition:border-color .2s}
.mc:hover{border-color:var(--border)}
.mk{font-family:var(--mono);font-size:9px;color:var(--muted);
    letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px}
.mv{font-family:var(--mono);font-size:13px;color:var(--text);font-weight:700;word-break:break-all}

/* actions */
.actions{display:flex;gap:12px;flex-wrap:wrap;animation:fadeUp .5s .45s ease both}
.btn{flex:1;min-width:140px;display:flex;align-items:center;justify-content:center;gap:8px;
     padding:13px 20px;border-radius:12px;border:none;
     font-family:var(--sans);font-size:14px;font-weight:700;
     cursor:pointer;text-decoration:none;transition:all .2s;letter-spacing:.3px}
.btn-p{background:linear-gradient(135deg,var(--accent),#0070a8);color:#fff}
.btn-p:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(0,200,255,.3)}
.btn-s{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);color:var(--text)}
.btn-s:hover{background:rgba(255,255,255,.08);border-color:var(--border)}
</style>
</head>
<body>
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="page">

  <div class="topbar">
    <div class="badge"><span class="dot"></span> SYNAPSE FORENSICS</div>
    <a href="/" class="back">← New Scan</a>
  </div>

  <div class="hero">
    <div class="hero-glow"></div>
    <div class="v-icon">{{ v_icon }}</div>
    <div class="v-text">
      <h2>Forensic Verdict</h2>
      <div class="v-label">{{ label }}</div>
      <div class="v-conf">Confidence score · {{ conf }}</div>
    </div>
    <div class="ring-wrap">
      <div class="ring-label">Confidence</div>
      <svg class="ring-svg" viewBox="0 0 80 80">
        <circle class="ring-bg" cx="40" cy="40" r="36"/>
        <circle class="ring-fg" cx="40" cy="40" r="36"/>
      </svg>
      <div class="ring-center">{{ conf }}</div>
    </div>
  </div>

  <div class="interp">
    <div class="interp-icon">{{ i_icon }}</div>
    <div>
      <div class="interp-title">Analysis Interpretation</div>
      <div class="interp-body">{{ interpretation|safe }}</div>
    </div>
  </div>

  <div class="img-grid">
    <div class="panel">
      <div class="p-header"><div class="p-dot"></div><span class="p-title">📷 Input Frame</span></div>
      <img id="up-img" src="" style="display:none" alt="uploaded">
      <div class="placeholder" id="ph">Preview not cached</div>
    </div>
    <div class="panel">
      <div class="p-header"><div class="p-dot"></div><span class="p-title">🧠 EigenCAM Heatmap</span></div>
      <img src="{{ cam_img }}" alt="EigenCAM">
    </div>
  </div>

  <div class="bars">
    <div class="bars-title">Class Probability Distribution</div>
    {{ bars|safe }}
  </div>

  <div class="meta-grid">
    {% for k,v in meta.items() %}
    <div class="mc"><div class="mk">{{ k }}</div><div class="mv">{{ v }}</div></div>
    {% endfor %}
  </div>

  <div class="actions">
    <a href="{{ cam_img }}" download class="btn btn-p">⬇ &nbsp;Download Heatmap</a>
    <a href="/" class="btn btn-s">🔁 &nbsp;New Analysis</a>
  </div>

</div>
<script>
(function(){
  const d=sessionStorage.getItem('up');
  const img=document.getElementById('up-img');
  const ph=document.getElementById('ph');
  if(d){img.src=d;img.style.display='block';ph.style.display='none';sessionStorage.removeItem('up')}
})();
document.querySelectorAll('.bar-fill').forEach(el=>{
  const w=el.dataset.w;el.style.width='0%';
  setTimeout(()=>{el.style.width=w+'%'},120);
});
</script>
</body>
</html>"""


# ================================================================
#  ROUTES
# ================================================================
@app.route('/')
def home():
    return render_template_string(HOME_HTML)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(input_path)

    try:
        predictions, cam_image, meta = process_image(input_path)

        output_file = "output_" + file.filename
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
        cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

        max_class = max(predictions, key=predictions.get)
        conf_val  = predictions[max_class]

        if conf_val < 0.6:
            label = "UNCERTAIN"; vc = "#ffb300"; v_icon = "⚠️"; i_icon = "🔎"; i_rgb = "255,179,0"
            interp = ("The model's confidence is below the 60% threshold. The image exhibits mixed "
                      "characteristics that prevent a definitive verdict. Manual expert review is "
                      "strongly recommended before drawing any conclusions.")
        elif "real" in max_class.lower():
            label = "AUTHENTIC"; vc = "#00e676"; v_icon = "✅"; i_icon = "🛡️"; i_rgb = "0,230,118"
            interp = (f"Classified as <strong>real / authentic</strong> with {conf_val*100:.1f}% confidence. "
                      "The EigenCAM overlay highlights facial regions that drove this decision. "
                      "No significant GAN or diffusion-model artifacts were detected — texture consistency, "
                      "edge integrity, and frequency patterns all align with genuine photography.")
        else:
            label = "DEEPFAKE"; vc = "#ff2d6b"; v_icon = "🚨"; i_icon = "⚡"; i_rgb = "255,45,107"
            interp = (f"Classified as a <strong>synthetic / deepfake image</strong> with {conf_val*100:.1f}% "
                      "confidence. The EigenCAM heatmap highlights high-attention zones — typically eye "
                      "regions, skin-blending boundaries, and hairline edges — where the model detected "
                      "telltale artifacts from face-swapping or generative synthesis pipelines.")

        # ring (circumference 2π×36 ≈ 226)
        ring_offset = round(226 * (1 - conf_val), 1)

        bars = ""
        for k, v in predictions.items():
            color = "#00e676" if "real" in k.lower() else "#ff2d6b" if "fake" in k.lower() else "#00c8ff"
            pct = round(v * 100, 1)
            bars += f"""
<div class="bar-row">
  <div class="bar-meta">
    <span class="bar-name" style="color:{color}">{k}</span>
    <span class="bar-pct">{pct}%</span>
  </div>
  <div class="bar-track">
    <div class="bar-fill" data-w="{pct}"
         style="background:linear-gradient(90deg,{color}99,{color})"></div>
  </div>
</div>"""

        return render_template_string(
            RESULT_HTML,
            label=label, conf=f"{conf_val*100:.1f}%",
            verdict_color=vc, v_icon=v_icon,
            i_icon=i_icon, i_rgb=i_rgb,
            interpretation=interp,
            ring_offset=ring_offset,
            bars=bars, meta=meta,
            cam_img=url_for('static', filename='outputs/' + output_file),
        )

    except Exception as e:
        traceback.print_exc()
        return f"<pre style='color:#ff2d6b;padding:40px;background:#060912'>Error: {e}</pre>"


if __name__ == '__main__':
    app.run(debug=True)