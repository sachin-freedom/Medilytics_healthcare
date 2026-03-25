# pdf_export.py
# Captures the entire rendered Streamlit page and saves it as a PDF file
# Uses html2canvas + jsPDF loaded from CDN — no pip packages needed

import streamlit as st
import streamlit.components.v1 as components


def render_pdf_button(filename: str = "Medilytics_Report"):
    """
    Renders a 'Save as PDF' button.
    When clicked:
      1. html2canvas captures the full Streamlit page as a canvas
      2. jsPDF converts it to a PDF
      3. Browser downloads it immediately as a .pdf file
    The filename is sanitised and .pdf is appended automatically.
    """
    safe_name = filename.replace(" ", "_").replace("/", "-")

    components.html(f"""
<!DOCTYPE html>
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<style>
  body {{ margin:0; padding:0; background:transparent; }}

  .pdf-btn {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: transparent;
    border: 1px solid #FFD700;
    color: #FFD700;
    font-family: 'Rajdhani', 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 7px 18px;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.2s ease, color 0.2s ease, transform 0.15s ease;
    user-select: none;
  }}

  .pdf-btn:hover {{
    background: #FFD700;
    color: #020817;
    transform: translateY(-1px);
  }}

  .pdf-btn:active {{ transform: translateY(0); }}

  .pdf-btn svg {{
    flex-shrink: 0;
  }}

  .pdf-btn.loading {{
    opacity: 0.7;
    pointer-events: none;
    cursor: wait;
  }}

  #status {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #94A3B8;
    margin-left: 10px;
    display: inline-block;
    vertical-align: middle;
    min-width: 160px;
  }}
</style>
</head>
<body>
<button class="pdf-btn" id="pdfBtn" onclick="savePDF()">
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
       stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="7 10 12 15 17 10"/>
    <line x1="12" y1="15" x2="12" y2="3"/>
  </svg>
  Save as PDF
</button>
<span id="status"></span>

<script>
async function savePDF() {{
  const btn    = document.getElementById('pdfBtn');
  const status = document.getElementById('status');

  btn.classList.add('loading');
  btn.textContent = 'Capturing...';
  status.textContent = 'Please wait — rendering page...';

  try {{
    // Target the Streamlit main content area in the parent window
    const parentDoc = window.parent.document;

    // Find the main content block
    const target =
      parentDoc.querySelector('[data-testid="stMainBlockContainer"]') ||
      parentDoc.querySelector('[data-testid="block-container"]') ||
      parentDoc.querySelector('.main') ||
      parentDoc.body;

    status.textContent = 'Capturing charts and content...';

    const canvas = await window.parent.html2canvas(target, {{
      scale: 2,                    // 2× resolution for crisp PDF
      useCORS: true,
      allowTaint: true,
      backgroundColor: window.getComputedStyle(target).backgroundColor || '#020817',
      logging: false,
      windowWidth: parentDoc.documentElement.scrollWidth,
      scrollX: 0,
      scrollY: 0,
      height: target.scrollHeight,
      width: target.scrollWidth,
    }});

    status.textContent = 'Generating PDF file...';

    const {{ jsPDF }} = window.parent.jspdf;

    const imgW  = canvas.width;
    const imgH  = canvas.height;

    // A4 page in mm
    const pageW = 297;
    const pageH = 210;

    // Scale to fit width, allow multi-page height
    const ratio    = pageW / imgW;
    const scaledH  = imgH * ratio;

    const pdf = new jsPDF({{
      orientation: scaledH > pageW ? 'portrait' : 'landscape',
      unit: 'mm',
      format: 'a4',
    }});

    const pdfW = pdf.internal.pageSize.getWidth();
    const pdfH = pdf.internal.pageSize.getHeight();

    // Slice image across pages
    const imgData   = canvas.toDataURL('image/jpeg', 0.95);
    const totalPages = Math.ceil((imgH * ratio) / pdfH);

    for (let i = 0; i < totalPages; i++) {{
      if (i > 0) pdf.addPage();

      const srcY  = i * (pdfH / ratio);
      const sliceH = Math.min(pdfH / ratio, imgH - srcY);

      // Create a temporary canvas for this page slice
      const sliceCanvas = document.createElement('canvas');
      sliceCanvas.width  = imgW;
      sliceCanvas.height = sliceH;
      const ctx = sliceCanvas.getContext('2d');
      ctx.drawImage(canvas, 0, srcY, imgW, sliceH, 0, 0, imgW, sliceH);

      const sliceData = sliceCanvas.toDataURL('image/jpeg', 0.95);
      pdf.addImage(sliceData, 'JPEG', 0, 0, pdfW, sliceH * ratio);
    }}

    pdf.save('{safe_name}.pdf');

    status.textContent = 'PDF saved successfully!';
    setTimeout(() => {{ status.textContent = ''; }}, 3000);

  }} catch (err) {{
    console.error('PDF error:', err);
    status.textContent = 'Error — try again.';
  }}

  // Restore button
  btn.classList.remove('loading');
  btn.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
    Save as PDF`;
}}

// Load html2canvas + jsPDF into PARENT window so they can capture parent DOM
function loadScript(src, globalName) {{
  return new Promise((resolve, reject) => {{
    if (window.parent[globalName]) {{ resolve(); return; }}
    const s = window.parent.document.createElement('script');
    s.src = src; s.onload = resolve; s.onerror = reject;
    window.parent.document.head.appendChild(s);
  }});
}}

// Inject libraries into parent on load
(async () => {{
  try {{
    await loadScript(
      'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js',
      'html2canvas'
    );
    await loadScript(
      'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js',
      'jspdf'
    );
  }} catch(e) {{
    console.warn('PDF libraries load error:', e);
  }}
}})();
</script>
</body>
</html>
""", height=52, scrolling=False)
