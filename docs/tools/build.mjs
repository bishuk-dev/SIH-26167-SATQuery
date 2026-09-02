import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sourcePath = resolve(root, "Beginner-Learning-Guide-Raw.md");
const processedPath = resolve(root, "Beginner-Learning-Guide.md");
const fragmentPath = resolve(root, "tmp/article.html");
const outputPath = resolve(root, "index.html");

const original = readFileSync(sourcePath, "utf8").replace(/\r\n/g, "\n");
const continuationMarker = "# Part XXXI — The part most beginner tutorials skip:";
const markerIndex = original.indexOf(continuationMarker);

if (markerIndex < 0) {
  throw new Error("Could not locate Part XXXI; source structure changed.");
}

const definitionPattern = /^\[(\d+)\]:\s+(\S+)(?:\s+"([^"]*)")?\s*$/gm;

function extractDefinitions(segment, offset = 0) {
  const definitions = [];
  const body = segment.replace(definitionPattern, (_, id, url, title = "") => {
    definitions.push({ id: String(Number(id) + offset), originalId: id, url, title });
    return "";
  });
  return { body, definitions };
}

const first = extractDefinitions(original.slice(0, markerIndex));
const second = extractDefinitions(original.slice(markerIndex), 11);

// The second source block restarts at [1]. Renaming only the hidden reference
// identifiers prevents incorrect links without changing any displayed prose.
let secondBody = second.body;
for (let id = 4; id >= 1; id -= 1) {
  secondBody = secondBody.replaceAll(`][${id}]`, `][${id + 11}]`);
}

const allDefinitions = [...first.definitions, ...second.definitions];

const figureInsertions = new Map([
  [
    "# 62. What is inside BigEarthNet.txt?",
    `<figure class="paper-figure" data-figure>
  <img src="assets/figures/bigearthnet-overview.png" alt="BigEarthNet.txt overview showing Sentinel-1 and Sentinel-2 imagery and its fifteen vision-language tasks" loading="lazy" decoding="async">
</figure>`
  ],
  [
    "# 65. How were BigEarthNet.txt captions created?",
    `<figure class="paper-figure paper-figure--portrait" data-figure>
  <img src="assets/figures/caption-generation.png" alt="BigEarthNet.txt caption generation process using reference maps, templates, paraphrasing, and refinement" loading="lazy" decoding="async">
</figure>`
  ],
  [
    "# Part XVII — Planner, Executor and Verifier",
    `<figure class="paper-figure paper-figure--wide" data-figure>
  <img src="assets/figures/agentic-eo-blueprint.png" alt="Agentic Earth observation blueprint with EO context, planner, executor, verifier, and structured state" loading="lazy" decoding="async">
</figure>`
  ],
  [
    "# Part XXXVI — Three ways to fuse modalities",
    `<figure class="paper-figure paper-figure--wide" data-figure>
  <img src="assets/figures/multimodal-gfm-architecture.png" alt="Multimodal geospatial foundation model architectures for feature alignment, fusion, and task-driven decoding" loading="lazy" decoding="async">
</figure>`
  ]
]);

function addFigures(markdown) {
  const output = [];
  let inFence = false;
  for (const line of markdown.split("\n")) {
    if (line.startsWith("```")) inFence = !inFence;
    output.push(line);
    if (!inFence && figureInsertions.has(line)) {
      output.push("", figureInsertions.get(line), "");
    }
  }
  return output.join("\n");
}

function normalizeHeadingLevels(markdown) {
  const output = [];
  let inFence = false;
  let isFirstHeading = true;

  for (const line of markdown.split("\n")) {
    if (line.startsWith("```")) {
      inFence = !inFence;
      output.push(line);
      continue;
    }

    if (inFence || !line.startsWith("#")) {
      output.push(line);
      continue;
    }

    if (isFirstHeading && line.startsWith("# ")) {
      isFirstHeading = false;
      output.push(line);
    } else if (/^# Part\b/.test(line) || line === "# Sources") {
      output.push(line.replace(/^# /, "## "));
    } else if (/^#{1,2} \d+\./.test(line)) {
      output.push(line.replace(/^#{1,2} /, "### "));
    } else if (/^#{2,3} /.test(line)) {
      output.push(line.replace(/^#{2,3} /, "#### "));
    } else {
      output.push(line);
    }
  }

  return output.join("\n");
}

const visibleSources = allDefinitions
  .map(({ url }) => `1. [${url}](${url})`)
  .join("\n");

const hiddenDefinitions = allDefinitions
  .map(({ id, url, title }) => `[${id}]: ${url}${title ? ` "${title}"` : ""}`)
  .join("\n");

let processed = `${first.body.trimEnd()}\n\n${secondBody.trim()}\n\n# Sources\n\n${visibleSources}\n\n${hiddenDefinitions}\n`;
processed = addFigures(processed);
processed = normalizeHeadingLevels(processed);

mkdirSync(resolve(root, "tmp"), { recursive: true });
writeFileSync(processedPath, processed, "utf8");

const pandoc = spawnSync(
  "pandoc",
  [
    processedPath,
    "--from=markdown+tex_math_dollars+raw_html",
    "--to=html5",
    "--mathjax",
    "--wrap=none",
    "--output",
    fragmentPath
  ],
  { encoding: "utf8" }
);

if (pandoc.status !== 0) {
  throw new Error(pandoc.stderr || "Pandoc failed.");
}

let article = readFileSync(fragmentPath, "utf8");

const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A beginner-friendly guide to satellite imagery, multimodal vision-language models, and agentic remote sensing for SatQuery AI.">
  <meta name="theme-color" content="#071b2e">
  <title>Beginner’s Guide to SatQuery AI</title>
  <link rel="stylesheet" href="./assets/styles.css">
  <script>
    window.MathJax = {
      tex: { inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]] },
      options: { skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"] }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <script defer src="./assets/script.js"></script>
</head>
<body>
  <div class="reading-progress" aria-hidden="true"><span id="reading-progress"></span></div>
  <header class="site-header">
    <a class="brand" href="#top" aria-label="SatQuery guide home">
      <span class="brand__mark" aria-hidden="true">SQ</span>
      <span>SatQuery Learning Guide</span>
    </a>
    <div class="site-header__actions">
      <button class="header-button" id="theme-toggle" type="button" aria-label="Switch color theme" title="Switch color theme">Theme</button>
      <button class="header-button menu-toggle" id="menu-toggle" type="button" aria-controls="sidebar" aria-expanded="false" aria-label="Toggle navigation menu">
        <svg class="menu-icon menu-icon--open" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16" /></svg>
        <svg class="menu-icon menu-icon--close" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 18 18 6M6 6l12 12" /></svg>
      </button>
    </div>
  </header>
  <div class="site-shell" id="top">
    <aside class="sidebar" id="sidebar" aria-label="Guide navigation">
      <div class="sidebar__inner">
        <p class="sidebar__label">Guide map</p>
        <nav id="toc-side" class="toc-side" aria-label="Guide sections"></nav>
      </div>
    </aside>
    <button class="drawer-backdrop" id="drawer-backdrop" type="button" aria-label="Close navigation"></button>
    <main class="main-content">
      <article class="prose" id="article">
${article}
      </article>
    </main>
  </div>
  <button class="back-to-top" id="back-to-top" type="button" aria-label="Back to top">↑</button>
  <dialog class="figure-dialog" id="figure-dialog">
    <button class="figure-dialog__close" id="figure-dialog-close" type="button" aria-label="Close image">Close</button>
    <img id="figure-dialog-image" alt="">
  </dialog>
</body>
</html>
`;

writeFileSync(outputPath, html, "utf8");
console.log(`Built ${outputPath}`);
