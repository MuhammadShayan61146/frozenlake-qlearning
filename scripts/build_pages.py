"""
Build the static, server-free site into docs/ for GitHub Pages.

    python scripts/build_pages.py [site-url]

The dashboard is the same frontend as the FastAPI version; the only difference is
that it loads local-engine.js (the in-browser Q-learning engine) instead of calling
the backend. Re-run this after changing anything in frontend/.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = (sys.argv[1] if len(sys.argv) > 1 else "https://muhammadshayan61146.github.io/frozenlake-qlearning/").rstrip("/") + "/"
TITLE = "Frozen Lake Lab: watch a Q-learning agent learn"
DESC = "Interactive reinforcement learning demo. Tune hyperparameters, retrain live and watch an agent learn to cross slippery ice. Runs entirely in your browser."

html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

meta = f"""  <meta property="og:type" content="website">
  <meta property="og:title" content="{TITLE}">
  <meta property="og:description" content="{DESC}">
  <meta property="og:url" content="{SITE}">
  <meta property="og:image" content="{SITE}og-image.png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta name="twitter:card" content="summary_large_image">
"""
assert '<link rel="stylesheet" href="static/app.css">' in html
html = html.replace('  <link rel="stylesheet" href="static/app.css">', meta + '  <link rel="stylesheet" href="static/app.css">', 1)
assert '<script src="static/app.js"></script>' in html
html = html.replace(
    '<script src="static/app.js"></script>',
    '<script>window.FL_LOCAL = true;</script>\n<script src="static/local-engine.js"></script>\n<script src="static/app.js"></script>',
    1,
)

out = ROOT / "docs"
(out / "static").mkdir(parents=True, exist_ok=True)
(out / "index.html").write_text(html, encoding="utf-8")
for name in ("app.css", "app.js", "local-engine.js"):
    shutil.copy(ROOT / "frontend" / "static" / name, out / "static" / name)
(out / ".nojekyll").write_text("")
print("built", out, "for", SITE)
