# SatQuery Beginner Learning Guide

This is a dependency-free static website generated from `Beginner-Learning-Guide.md`.

## Preview locally

Open `index.html` directly, or run any static file server in this directory.

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Deploy with GitHub Pages

1. Upload the complete folder to a GitHub repository.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the branch containing these files and the `/ (root)` folder.
5. Save. GitHub will provide the public URL after deployment finishes.

## Rebuild after editing the Markdown

The checked-in `index.html` is already complete. If you update the Markdown, install Pandoc and run:

```bash
node tools/build.mjs
```

The site uses only `index.html`, `styles.css`, `script.js`, and the images under `assets/` at runtime.
