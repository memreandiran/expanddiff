# ExpandDiff — project page

Static project page for *ExpandDiff: Dynamic Range Expanding Diffusion for
Single-Image HDR Reconstruction*. Plain HTML with no build step and no
dependencies: GitHub Pages serves `index.html` as-is.

## Layout

    index.html   the whole page, styling included
    assets/      teaser images (512x512 PNG)
    .nojekyll    serve the files verbatim, without Jekyll processing

## Filling in the links

The remaining buttons in the header start disabled (Code and Checkpoints are live). Each is marked with a
`PLACEHOLDER` comment in `index.html`. To activate one, set its `href` and
remove `disabled` from its `class`:

| button | href to use |
|---|---|
| arXiv | `https://arxiv.org/abs/XXXX.XXXXX` |
| Paper | `paper.pdf`, uploaded next to `index.html` |
| Supplementary | `supplementary.pdf`, uploaded next to `index.html` |

## Local preview

    python3 -m http.server 8000

then open <http://localhost:8000>.
