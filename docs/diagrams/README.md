# Offline homepage diagrams

Pre-rendered SVGs for the comparison site index. Sources are the `*.mmd` files;
`tools/build_site.py` copies the SVGs into `docs/site/diagrams/`.

Regenerate (needs `@mermaid-js/mermaid-cli` + a browser for Puppeteer):

```bash
mmdc -i docs/diagrams/arch.mmd -o docs/diagrams/arch.svg -b transparent
mmdc -i docs/diagrams/concepts.mmd -o docs/diagrams/concepts.svg -b transparent
python tools/build_site.py
```

Do not load Mermaid from a CDN; the published site must work offline.
