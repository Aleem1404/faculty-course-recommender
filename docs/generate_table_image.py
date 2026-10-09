import os
import subprocess
from pathlib import Path

# Paths
docs_dir = Path(r"F:\Dissertation-new\faculty-course-recommender\docs")
docs_new_dir = Path(r"F:\Dissertation-new\faculty-course-recommender\docs-new")
web_charts_dir = Path(r"F:\Dissertation-new\faculty-course-recommender\web\data\charts")

for d in [docs_dir, docs_new_dir, web_charts_dir]:
    d.mkdir(parents=True, exist_ok=True)

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #ffffff;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 24px;
    color: #334155;
    -webkit-font-smoothing: antialiased;
  }

  .table-container {
    width: 1080px;
    background: #ffffff;
    padding: 20px 24px 16px 24px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
  }

  th {
    text-align: left;
    font-size: 15px;
    font-weight: 600;
    color: #64748b;
    padding: 12px 14px 14px 14px;
    border-bottom: 2px solid #e2e8f0;
    letter-spacing: 0.01em;
  }

  th:nth-child(1) { width: 150px; }
  th:nth-child(2) { width: 230px; }
  th:nth-child(3) { width: 230px; }
  th:nth-child(4) { width: 230px; }
  th:nth-child(5) { width: 240px; }

  td {
    padding: 18px 14px;
    font-size: 13.5px;
    line-height: 1.48;
    color: #334155;
    vertical-align: middle;
    border-bottom: 1px solid #f1f5f9;
  }

  tr:last-child td {
    border-bottom: none;
  }

  /* Model Column with Left Accent Bar */
  .model-cell {
    position: relative;
    padding-left: 18px;
  }

  .accent-bar {
    position: absolute;
    left: 0;
    top: 10px;
    bottom: 10px;
    width: 4.5px;
    border-radius: 3px;
  }

  .bar-matching { background-color: #2563eb; }
  .bar-explainability { background-color: #eab308; }
  .bar-collaboration { background-color: #ec4899; }
  .bar-allocation { background-color: #14b8a6; }

  .model-name {
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
    display: block;
    margin-bottom: 3px;
  }

  .model-subtitle {
    font-size: 12.5px;
    font-weight: 500;
    color: #64748b;
  }

  .arrow-bullet {
    color: #94a3b8;
    margin-right: 4px;
    font-weight: 600;
  }

  .input-text {
    color: #334155;
  }

  /* Footer Legend */
  .legend-container {
    display: flex;
    align-items: center;
    gap: 28px;
    margin-top: 18px;
    padding-top: 16px;
    border-top: 1px solid #e2e8f0;
    font-size: 13px;
    font-weight: 500;
    color: #64748b;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .legend-pill {
    width: 20px;
    height: 4.5px;
    border-radius: 2px;
  }
</style>
</head>
<body>

<div class="table-container" id="report-table">
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Input</th>
        <th>Purpose</th>
        <th>Output</th>
        <th>Evaluation</th>
      </tr>
    </thead>
    <tbody>
      <!-- M0 -->
      <tr>
        <td class="model-cell">
          <div class="accent-bar bar-matching"></div>
          <span class="model-name">M0</span>
          <span class="model-subtitle">Raw TF-IDF</span>
        </td>
        <td>
          <div class="input-text">Original module<br><span class="arrow-bullet">&rarr;</span> text and staff profiles</div>
        </td>
        <td>Create a transparent lexical baseline</td>
        <td>Ranked staff candidates</td>
        <td>Case study; NDCG at 5, precision, MRR and MAP</td>
      </tr>

      <!-- M1 -->
      <tr>
        <td class="model-cell">
          <div class="accent-bar bar-matching"></div>
          <span class="model-name">M1</span>
          <span class="model-subtitle">Enriched TF-IDF</span>
        </td>
        <td>
          <div class="input-text">M0 data plus<br><span class="arrow-bullet">&rarr;</span> publication metadata</div>
        </td>
        <td>Test whether enrichment improves lexical matching</td>
        <td>Updated ranked candidates</td>
        <td>Change from M0; benchmark ranking metrics</td>
      </tr>

      <!-- M2 -->
      <tr>
        <td class="model-cell">
          <div class="accent-bar bar-matching"></div>
          <span class="model-name">M2</span>
          <span class="model-subtitle">Semantic model</span>
        </td>
        <td>
          <div class="input-text">Cleaned module<br><span class="arrow-bullet">&rarr;</span> text and enriched staff profiles</div>
        </td>
        <td>Capture contextual similarity beyond keywords</td>
        <td>Semantic ranking; M2-H eligibility gate</td>
        <td>Coverage, rejection audit and benchmark metrics</td>
      </tr>

      <!-- M3-KG -->
      <tr>
        <td class="model-cell">
          <div class="accent-bar bar-explainability"></div>
          <span class="model-name">M3-KG</span>
          <span class="model-subtitle">Graph hybrid</span>
        </td>
        <td>
          <div class="input-text">M2-H candidates<br><span class="arrow-bullet">&rarr;</span> plus topic and organisational graph</div>
        </td>
        <td>Rerank candidates and explain recommendations</td>
        <td>Reranked list and evidence paths</td>
        <td>Rank changes, path coverage, topic evidence and NDCG</td>
      </tr>

      <!-- M4 -->
      <tr>
        <td class="model-cell">
          <div class="accent-bar bar-collaboration"></div>
          <span class="model-name">M4</span>
          <span class="model-subtitle">Collaboration</span>
        </td>
        <td>
          <div class="input-text">Canonical staff<br><span class="arrow-bullet">&rarr;</span> and module topic profiles</div>
        </td>
        <td>Find complementary cross-department staff pairs</td>
        <td>Ranked primary and guest-collaborator pairs</td>
        <td>Combined topic coverage, complementarity gain and retained pairs</td>
      </tr>

      <!-- M5 -->
      <tr>
        <td class="model-cell">
          <div class="accent-bar bar-allocation"></div>
          <span class="model-name">M5</span>
          <span class="model-subtitle">Capacity-aware</span>
        </td>
        <td>
          <div class="input-text">Candidate lists<br><span class="arrow-bullet">&rarr;</span> plus simulated workload constraints</div>
        </td>
        <td>Reduce assignment concentration while retaining relevance</td>
        <td>Simulated balanced allocation</td>
        <td>Gini, utilisation, maximum load and relevance retention</td>
      </tr>
    </tbody>
  </table>

  <div class="legend-container">
    <div class="legend-item">
      <div class="legend-pill bar-matching"></div>
      <span>Matching</span>
    </div>
    <div class="legend-item">
      <div class="legend-pill bar-explainability"></div>
      <span>Explainability</span>
    </div>
    <div class="legend-item">
      <div class="legend-pill bar-collaboration"></div>
      <span>Collaboration</span>
    </div>
    <div class="legend-item">
      <div class="legend-pill bar-allocation"></div>
      <span>Allocation</span>
    </div>
  </div>
</div>

</body>
</html>
"""

html_path = docs_dir / "table_models_overview.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Wrote HTML template to {html_path}")

# Render with Edge Headless
edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_path):
    edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

temp_png = docs_dir / "temp_render.png"
out_png_1 = docs_dir / "Table_3.1_System_Models_Overview.png"
out_png_2 = docs_new_dir / "Table_3.1_System_Models_Overview.png"
out_png_3 = web_charts_dir / "Table_3.1_System_Models_Overview.png"
out_pdf_1 = docs_dir / "Table_3.1_System_Models_Overview.pdf"
out_pdf_2 = docs_new_dir / "Table_3.1_System_Models_Overview.pdf"

# Headless screenshot with high DPI scale factor
cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    "--force-device-scale-factor=3",
    "--window-size=1200,900",
    "--hide-scrollbars",
    f"--screenshot={temp_png}",
    html_path.as_uri()
]

print("Rendering high-res screenshot with Edge...")
subprocess.run(cmd, check=True)

from PIL import Image, ImageChops

# Crop whitespace automatically
img = Image.open(temp_png)
bg = Image.new(img.mode, img.size, (255, 255, 255, 255))
diff = ImageChops.difference(img, bg)
bbox = diff.getbbox()

if bbox:
    # Add modest padding (e.g. 40px at 3x scale)
    pad = 40
    crop_box = (
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(img.width, bbox[2] + pad),
        min(img.height, bbox[3] + pad)
    )
    cropped = img.crop(crop_box)
else:
    cropped = img

# Save PNGs with 300 DPI metadata
for p in [out_png_1, out_png_2, out_png_3]:
    cropped.save(p, "PNG", dpi=(300, 300))
    print(f"Saved High-Res Image (300 DPI): {p} (Resolution: {cropped.width}x{cropped.height})")

# Also save as high-res PDF for LaTeX / reports
rgb_img = cropped.convert('RGB')
for pdf_p in [out_pdf_1, out_pdf_2]:
    rgb_img.save(pdf_p, "PDF", resolution=300.0)
    print(f"Saved High-Res Vector/Raster PDF: {pdf_p}")

if temp_png.exists():
    temp_png.unlink()

print("Done successfully!")

