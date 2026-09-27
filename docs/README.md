# ExpandDiff — project page

Static project page for *ExpandDiff: Dynamic Range Expanding Diffusion for
Single-Image HDR Reconstruction*. Plain HTML with no build step and no
dependencies: GitHub Pages serves `index.html` as-is.

## Layout

    index.html   the whole page, styling included
    assets/      teaser images (512x512 PNG)
    .nojekyll    serve the files verbatim, without Jekyll processing

## Local preview

    python3 -m http.server 8000

then open <http://localhost:8000>.
